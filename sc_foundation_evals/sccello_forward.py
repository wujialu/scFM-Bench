## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.
import os
import sys
import importlib.util
import pickle
from tqdm import tqdm
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import trange
from datasets import Dataset, load_from_disk
from . import utils
from .data import InputData
from .helpers.custom_logging import log

from geneformer.tokenizer import TranscriptomeTokenizer
from datasets import load_from_disk


from .langcell_modules import DataCollatorForCellClassificationWithCLS as DataCollatorForCellClassification
# from .sccello.src.collator.collator_for_classification import DataCollatorForCellClassification
from .sccello.src.model_prototype_contrastive import PrototypeContrastiveForMaskedLM as scCelloModel


import warnings
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")


class scCello_instance():
    def __init__(self,
                 saved_model_path: Optional[str] = None,
                 model_run: str = "pretrained",
                 model_files: Dict[str, str] = None,
                 batch_size: int = 8,
                 save_dir: Optional[str] = None, 
                 explicit_save_dir: bool = False,
                 num_workers: int = 0,
                 log_wandb: bool = False,
                 project_name: str = "scCello_eval",
                 add_cls: bool = True,
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
        self.add_cls = add_cls

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
        self.model = scCelloModel.from_pretrained(self.saved_model_path, output_hidden_states=True)
        self.model.to(self.device)
        
        log.info(f"Model successfully loaded from {self.saved_model_path}")
    
    def load_tokenized_dataset(self,
                               dataset_path: str,
                               cell_type_col: str = "cell_type") -> None:
        """
        The tokenized dataset is shared across Geneformer, LangCell, and scCello".
        Please run Geneformer first.
        """
        
        self.tokenized_dataset = load_from_disk(dataset_path)
        
        types = list(set(self.tokenized_dataset[cell_type_col]))
        # type2text = json.load(open(os.path.join(dataset_path, 'type2text.json')))
        # self.texts = [type2text[typename] for typename in types]
        type2num = dict([(type, i) for i, type in enumerate(types)])
        
        def classes_to_ids(example, idx):
            example["label"] = type2num[example[cell_type_col]]
            example["idx"] = idx
            return example
    
        self.tokenized_dataset = self.tokenized_dataset.map(classes_to_ids, with_indices=True, num_proc=16)
        remove_columns = self.tokenized_dataset.column_names
        columns_to_keep = ['input_ids', 'label', 'idx', "sorted_indices"]
        remove_columns = [col for col in remove_columns if col not in columns_to_keep]

        self.tokenized_dataset = self.tokenized_dataset.remove_columns(remove_columns)
        
    def tokenize_data(self,
                      adata_path: str,
                      dataset_path: str,
                      cell_type_col: str = "cell_type",
                      columns_to_keep: List[str] = ["adata_order"],
                      data_is_raw: bool = True,
                      include_zero_genes: bool = False):
        
        dataset_name = os.path.basename(adata_path).split(".")[0]
        tokenized_data_path = os.path.join(dataset_path, f"{dataset_name}.dataset")
        
        if not os.path.exists(tokenized_data_path):
            log.info(f"Tokenizing data from {adata_path} to {tokenized_data_path}")
            
            cols_to_keep = dict(zip([cell_type_col] + columns_to_keep, 
                                    ['cell_type'] + columns_to_keep))
                
            # initialize tokenizer
            self.tokenizer = TranscriptomeTokenizer(cols_to_keep, 
                                                    nproc = self.num_workers,
                                                    data_is_raw = data_is_raw,
                                                    include_zero_genes = include_zero_genes)

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
            
            # get the top directory of the adata_pathx
            adata_dir = os.path.dirname(adata_path)

            self.tokenizer.tokenize_data(adata_dir,
                                         dataset_path, 
                                         dataset_name,
                                         file_format=ext)

        # tokenizer does not return the dataset
        # load the dataset
        self.load_tokenized_dataset(tokenized_data_path)

    def load_vocab(self,
                   dict_paths: str) -> None:
        
        token_dictionary_path = os.path.join(dict_paths,
                                            "token_dictionary.pkl")
        with open(token_dictionary_path, "rb") as f:
            self.vocab = pickle.load(f) #!!! 25426 tokens, 25424 coding_miRNA_genes + <pad> + <mask>
        
        self.pad_token_id = self.vocab.get("<pad>")
        
        # size of vocabulary
        self.vocab_size = len(self.vocab)  

        gene_name_id_path = os.path.join(dict_paths,
                                        "gene_name_id_dict.pkl")
        with open(gene_name_id_path, "rb") as f:
            self.gene_name_id = pickle.load(f) #!!! 40248 genes with ensembl IDs and names
        self.id2name = {v: k for k, v in self.gene_name_id.items()}

        gene_median_file = os.path.join(dict_paths,
                                        "gene_median_dictionary.pkl")
        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)

        # gene keys for full vocabulary
        self.gene_keys = list(self.gene_median_dict.keys())

        # protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

    def get_dataloader(self):
        collator = DataCollatorForCellClassification(add_cls=self.add_cls)
        self.dataloader = DataLoader(self.tokenized_dataset, batch_size=self.batch_size, 
                                     collate_fn=collator, shuffle=False)
        
    def cellrepr_post_process(self, h_data, pass_cell_cls=False, normalize=False):
        if pass_cell_cls and normalize:
            h_data = self.model.cell_cls(h_data)[:, 0]
        
        else:
            h_data = h_data[:, 0]
            if normalize:
                h_data = F.normalize(h_data, p=2, dim=-1)

        return h_data

    def cell_encode(self, cell_input_ids, cell_atts, output_attentions=False, pass_cell_cls=True, normalize=True):
        cell = self.model.bert(input_ids=cell_input_ids.to(self.device), 
                               attention_mask=cell_atts.to(self.device),
                               output_attentions=output_attentions)
        cell_last_h = cell.last_hidden_state
        cell_emb = self.cellrepr_post_process(cell_last_h, pass_cell_cls=pass_cell_cls, normalize=normalize)

        if output_attentions:
            return cell_emb, cell.attentions

        else:
            return cell_emb
    
    def extract_embeddings(self,
                           data: InputData,
                           pass_cell_cls: bool = True,
                           normalize: bool = True,
                           embedding_key: str = "X_scCello",
                           ):

        # check if data loader is created
        if not self._check_attr("dataloader"):
            self.get_dataloader()

        # save the embeddings to subdir
        embeddings_subdir = os.path.join(self.output_dir, "model_outputs")
        os.makedirs(embeddings_subdir, exist_ok=True)

        cell_embeddings = []
        self.model.eval()
        for batch_id, batch_data in enumerate(tqdm(self.dataloader, desc="scCello (extracting embeddings)")):
            with torch.no_grad():
                #! mask zero padded genes (maxlen=2048)
                cellemb = self.cell_encode(batch_data['input_ids'], batch_data['attention_mask'],
                                           pass_cell_cls=pass_cell_cls, normalize=normalize)
            cell_embeddings.append(cellemb.detach().cpu().numpy())
            torch.cuda.empty_cache()

        self.cell_embeddings = np.concatenate(cell_embeddings, axis=0)

        # add embeddings to adata
        data.adata.obsm[embedding_key] = self.cell_embeddings

        # for plotting later, save the data.adata.obs
        # order here agrees with the order of the embeddings
        data.adata.obs.to_csv(os.path.join(embeddings_subdir, 
                                           "adata_obs.csv"))
        
    def extract_attn_weights(self, data: InputData, layer: int = -1):

        # check if data loader is created
        if not self._check_attr("dataloader"):
            self.get_dataloader()

        # save the embeddings to subdir
        os.makedirs(self.output_dir, exist_ok=True)

        dict_sum_condition = {}
        condition_ids = np.array(data.adata.obs["condition"].tolist())

        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in data.adata.var["ensembl_id"]]
        )[0]
        ori_gene_ids = np.array(
                [self.vocab[g] for g in data.adata.var["ensembl_id"][coding_miRNA_loc]]
            )
        ori_gene_names = [self.id2name.get(i, "") for i in data.adata.var["ensembl_id"][coding_miRNA_loc]]

        self.model.eval()
        for batch_id, batch_data in enumerate(tqdm(self.dataloader, desc="scCello (extracting attention weights)")):
            with torch.no_grad():
                # batch_data: dict_keys(['input_ids', 'attention_mask', 'labels', "sorted_indices"])
                _, attn_scores = self.cell_encode(batch_data['input_ids'], batch_data['attention_mask'], output_attentions=True)
                
            attn_scores = attn_scores[layer] # last layer
            num_heads = attn_scores.size(1)
            assert attn_scores.shape[-1] == batch_data["sorted_indices"].shape[-1] == batch_data["input_ids"].shape[-1]
            if self.add_cls:
                attn_scores = attn_scores[..., 1:, 1:]
                batch_data["sorted_indices"] = batch_data["sorted_indices"][:, 1:]
                batch_data["input_ids"] = batch_data["input_ids"] [:, 1:]
            M = attn_scores.shape[-1]

            # Rank normalization by row
            attn_scores = attn_scores.reshape((-1, M))
            order = torch.argsort(attn_scores, dim=1)
            rank = torch.argsort(order, dim=1)
            attn_scores = rank.reshape((-1, num_heads, M, M))/M
            # Rank normalization by column
            attn_scores = attn_scores.permute(0, 1, 3, 2).reshape((-1, M))
            order = torch.argsort(attn_scores, dim=1)
            rank = torch.argsort(order, dim=1)
            attn_scores = (rank.reshape((-1, num_heads, M, M))/M).permute(0, 1, 3, 2)
            # Average over attention heads
            attn_scores = attn_scores.mean(1)

            # Sort attention scores by original order
            sorted_indices = batch_data["sorted_indices"]
            attn_scores = utils.reverse_permute(attn_scores, sorted_indices) # # [batch_size, M, M]
            outputs = attn_scores.detach().cpu().numpy()

            gene_ids = batch_data["input_ids"]
            gene_ids = utils.reverse_permute(gene_ids, sorted_indices).numpy()
            assert np.all(gene_ids == ori_gene_ids, axis=1).all()
                
            batch_idx = batch_data["idx"].numpy()
            batch_conditions = condition_ids[batch_idx]
            for index, c in enumerate(batch_conditions):
                # Keep track of sum per condition
                if c not in dict_sum_condition:
                    dict_sum_condition[c] = outputs[index, :, :]
                else:
                    dict_sum_condition[c] += outputs[index, :, :] 

            torch.cuda.empty_cache()
        
        # Average rank-normed attention weights by condition
        dict_sum_condition_mean = dict_sum_condition.copy()
        dict_sum_condition_mean["gene_ids"] = ori_gene_ids
        dict_sum_condition_mean["gene_names"] = ori_gene_names
        groups = data.adata.obs.groupby('condition').groups
        for i in groups.keys():
            dict_sum_condition_mean[i] = dict_sum_condition_mean[i]/len(groups[i])
            assert dict_sum_condition_mean[i].shape == (len(ori_gene_names), len(ori_gene_names))

        with open(os.path.join(self.output_dir, 'pretrain_attn_condition_mean.pkl'), 'wb') as f:
            pickle.dump(dict_sum_condition_mean, f)