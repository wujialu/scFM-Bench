## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.
import os

import importlib.util
import pickle

from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from geneformer import EmbExtractor
from tqdm.auto import trange
from datasets import Dataset, load_from_disk
from . import utils
from .data import InputData
from .helpers.custom_logging import log

from geneformer.tokenizer import TranscriptomeTokenizer
from datasets import load_from_disk

import json
from transformers import BertTokenizer, BertModel # cell_bert
from .langcell_utils import BertModel as MedBertModel # text_bert
from .langcell_utils import LangCellDataCollatorForCellClassification as DataCollatorForCellClassification
from tqdm import tqdm


import warnings
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")

class Pooler(nn.Module):
    def __init__(self, config, pretrained_proj, proj_dim):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj))
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output


class Langcell_instance():
    def __init__(self,
                 saved_model_path: Optional[str] = None,
                 saved_tokenizer_path: Optional[str] = None,
                 model_run: str = "pretrained",
                 model_files: Dict[str, str] = {
                     "cell_encoder": "cell_bert",
                     "cell_proj": "cell_proj.bin",
                     "text_encoder": "text_bert",
                     "text_proj": "text_proj.bin",
                     "ctm_head": "ctm_head.bin",
                     },                 
                 batch_size: int = 8,
                 save_dir: Optional[str] = None, 
                 explicit_save_dir: bool = False,
                 num_workers: int = 0,
                 log_wandb: bool = False,
                 project_name: str = "Langcell_eval",
                 ) -> None:
        
        # check if the model run is supported
        supported_model_runs = ["pretrained"] #, "random", "finetune", "train"]
        if model_run not in supported_model_runs:
            msg = f"model_run must be one of {supported_model_runs}"
            log.error(msg)
            raise ValueError(msg)
        self.model_run = model_run

        self.saved_model_path = saved_model_path
        self.model_files = model_files
        self.saved_tokenizer_path = saved_tokenizer_path

        if num_workers == -1:
            num_workers = len(os.sched_getaffinity(0))

        if num_workers == 0:
            num_workers = 1

        self.num_workers = num_workers
        self.batch_size = batch_size

        # check if output directory exists
        if save_dir is not None:
            if explicit_save_dir:
                self.output_dir = save_dir
            else:
                self.output_dir = os.path.join(save_dir,
                                               self.run_id)
                # if the top out directory does not exist, create it
                if not os.path.exists(save_dir):
                    log.warning(f"Creating the top output directory {save_dir}")
                    os.makedirs(save_dir)
        else:
            # save in a current path
            self.output_dir = os.path.join(os.getcwd(), self.run_id)

        # if the out directory already exists, raise an error
        if os.path.exists(self.output_dir) and not explicit_save_dir:
            msg = f"Output directory: {self.output_dir} exists. Something is wrong!"
            log.error(msg)
            raise ValueError(msg)
        
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = torch.device("cuda" 
                                   if torch.cuda.is_available()
                                   else "cpu")
        
        log.info(f"Using device {self.device}")

        self.project_name = project_name
        if log_wandb:
            has_wandb = importlib.util.find_spec("wandb") is not None
            if not has_wandb:
                msg = "Wandb is not installed. Please install wandb to log to wandb."
                log.error(msg)
                raise RuntimeError(msg)
            if has_wandb:
                import wandb
            self._wandb = wandb
        else: 
            self._wandb = None

        # update this when saved config so that when training it only is saved once
        self.config_saved = False

    def _check_attr(self, 
                    attr: str, 
                    not_none: bool = True) -> bool:
        """
        Check if the argument is in the class
        """
        out = hasattr(self, attr)
        if not_none and out:
            out = getattr(self, attr) is not None
        return out

    def load_pretrained_model(self) -> None:
        
        self.model = BertModel.from_pretrained(os.path.join(self.saved_model_path, self.model_files["cell_encoder"]))
        self.model.pooler = Pooler(self.model.config, proj_dim=256,
                                   pretrained_proj=os.path.join(self.saved_model_path, self.model_files["cell_proj"]))
        self.model.to(self.device)
        
        self.text_encoder = MedBertModel.from_pretrained(os.path.join(self.saved_model_path, self.model_files["text_encoder"]), add_pooling_layer=True)
        self.text_encoder.pooler = Pooler(self.text_encoder.config, proj_dim=256,
                                          pretrained_proj=os.path.join(self.saved_model_path, self.model_files["text_proj"]))
        self.text_encoder.to(self.device)

        self.ctm_head = nn.Linear(self.text_encoder.config.hidden_size, 2)
        self.ctm_head.load_state_dict(torch.load(os.path.join(self.saved_model_path, self.model_files["ctm_head"])))
        self.ctm_head.to(self.device)
        
        log.info(f"Model successfully loaded from {self.saved_model_path}")
    
    def load_tokenizer(self) -> None:
        
        self.tokenizer = BertTokenizer.from_pretrained(self.saved_tokenizer_path)
        self.tokenizer.add_special_tokens({'bos_token':'[DEC]'})
        self.tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
        self.tokenizer.enc_token_id = self.tokenizer.additional_special_tokens_ids[0]  
    
    def load_tokenized_dataset(self,
                               dataset_path: str,
                               cell_type_col: str = "cell_type") -> None:
        
        self.tokenized_dataset = load_from_disk(dataset_path)
        
        types = list(set(self.tokenized_dataset[cell_type_col]))
        # type2text = json.load(open(os.path.join(dataset_path, 'type2text.json')))
        # self.texts = [type2text[typename] for typename in types]
        type2num = dict([(type, i) for i, type in enumerate(types)])
        
        def classes_to_ids(example):
            example["label"] = type2num[example[cell_type_col]]
            return example
    
        self.tokenized_dataset = self.tokenized_dataset.map(classes_to_ids, num_proc=16)
        remove_columns = self.tokenized_dataset.column_names
        remove_columns.remove('input_ids')
        remove_columns.remove('label')
        self.tokenized_dataset = self.tokenized_dataset.remove_columns(remove_columns)
        

    def tokenize_data(self,
                      adata_path: str,
                      dataset_path: str,
                      cell_type_col: str = "cell_type",
                      columns_to_keep: List[str] = ["adata_order"]):
        
        dataset_name = os.path.basename(adata_path).split(".")[0]
        
        cols_to_keep = dict(zip([cell_type_col] + columns_to_keep, 
                                ['cell_type'] + columns_to_keep))
        
        # initialize tokenizer
        self.tokenizer = TranscriptomeTokenizer(cols_to_keep, 
                                                nproc = self.num_workers)
        
        # get the extension from adata_path
        _, ext = os.path.splitext(adata_path)
        ext = ext.strip(".")

        if ext not in ["loom", "h5ad"]:
            msg = f"adata_path must be a loom or h5ad file. Got {ext}"
            log.error(msg)
            raise ValueError(msg)
        
        if ext == "h5ad":
            msg = ("using h5ad file. This sometimes causes issues. "
                   "If not working try with loom.")
            log.warning(msg)
        
        # get the top directory of the adata_path
        adata_dir = os.path.dirname(adata_path)

        self.tokenizer.tokenize_data(adata_dir,
                                     dataset_path, 
                                     dataset_name,
                                     file_format=ext)

        
        # tokenizer does not return the dataset
        # load the dataset
        self.load_tokenized_dataset(os.path.join(dataset_path, 
                                                 f"{dataset_name}.dataset"))
    
    
    def get_dataloader(self):
        collator = DataCollatorForCellClassification()
        self.dataloader = DataLoader(self.tokenized_dataset, batch_size=self.batch_size, 
                                     collate_fn=collator, shuffle=False)
    
    def text_encode(self, text):
        text = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        text = self.text_encoder(**text).pooler_output
        # text = F.normalize(model.text_projector(text))
        return text

    def cell_encode(self, cell_input_ids, cell_atts):
        cell = self.model(cell_input_ids.to(self.device), cell_atts.to(self.device))
        cell_last_h = cell.last_hidden_state
        cell_pooler = cell.pooler_output
        return cell_last_h, cell_pooler

    def ctm(self, text, cell_emb, cell_atts):
        text = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        output = self.text_encoder(**text,
                    encoder_hidden_states = cell_emb.to(self.device),
                    encoder_attention_mask = cell_atts.to(self.device),
                    return_dict = True,
                    mode = 'multimodal',
                    )
        logits = self.ctm_head(output.last_hidden_state[:, 0, :])
        logits = F.softmax(logits, dim=-1)[..., 1] # [n]
        return logits
    
    def extract_embeddings(self,
                           data: InputData,
                           embedding_key: str = "langcell"
                           ):

        # check if data loader is created
        if not self._check_attr("dataloader"):
            self.get_dataloader()

        # save the embeddings to subdir
        embeddings_subdir = os.path.join(self.output_dir, "model_outputs")
        os.makedirs(embeddings_subdir, exist_ok=True)

        cell_embeddings = []
        self.model.eval()
        for batch_id, batch_data in enumerate(tqdm(self.dataloader, desc="LangCell (extracting embeddings)")):
            with torch.no_grad():
                cell_last_h, cellemb = self.cell_encode(batch_data['input_ids'], batch_data['attention_mask'])
            cell_embeddings.append(cellemb.detach().cpu().numpy())
            torch.cuda.empty_cache()

        self.cell_embeddings = np.concatenate(cell_embeddings, axis=0)

        # add embeddings to adata
        data.adata.obsm[embedding_key] = self.cell_embeddings

        # for plotting later, save the data.adata.obs
        # order here agrees with the order of the embeddings
        data.adata.obs.to_csv(os.path.join(embeddings_subdir, 
                                           "adata_obs.csv"))
        
        
        