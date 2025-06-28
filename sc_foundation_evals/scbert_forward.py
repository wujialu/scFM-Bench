## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.
import os
import json
import time
import importlib.util

from typing import Dict, Optional, List, Union

from scipy.sparse import issparse
import torch
import scanpy as sc
from scanpy.get import _get_obs_rep, _set_obs_rep
from anndata import AnnData
import numpy as np
from torch.utils.data import Dataset, DataLoader

from . import utils
from .data import InputData
from .performer_pytorch import PerformerLM
import anndata as ad
from scipy import sparse
import random
from .helpers.custom_logging import log
from scgpt.utils import set_seed


class SCDataset(Dataset):
    def __init__(self, data, CLASS):
        super().__init__()
        self.data = data
        self.CLASS = CLASS

    def __getitem__(self, index):
        # rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[index].toarray()[0]
        full_seq[full_seq > (self.CLASS - 2)] = self.CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0])))
        return full_seq

    def __len__(self):
        return self.data.shape[0]
        

class scBERT_instance():
    def __init__(self,
                 saved_model_path: Optional[str] = None,
                 model_files: Dict[str, str] = {
                    "model_weights": "panglao_pretrain.pth", 
                    "ref_data": "panglao_human.h5ad",
                    "pos_emb_file": "gene2vec_16906.npy"
                 },
                 batch_size: int = 8,
                 save_dir: Optional[str] = None, 
                 explicit_save_dir: bool = False,
                 num_workers: int = 0,
                 n_log_reports: int = 10,
                 log_wandb: bool = False,
                 project_name: str = "scBERT_eval",
                 ) -> None:
    
        self.saved_model_path = saved_model_path
        self.model_files = model_files

        if batch_size % 8 != 0:
            batch_size_ = batch_size
            batch_size = (batch_size // 8 + 1) * 8
    
            msg = ("Using AMP by default (currently hardcoded) "
                   f"batch_size must be a multiple of 8 "
                   f"provided {batch_size_}, changing to {batch_size}")
            log.warning(msg)
        
        self.batch_size = batch_size
        
        self.run_id = (f'{time.strftime("%Y-%m-%d_%H-%M-%S")}')

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

        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device {self.device}")

        self.num_workers = num_workers
        self.n_log_reports = n_log_reports

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

    def create_configs(self, 
                       seed: int = 42, 
                       gene_num: int = 1000, 
                       bin_num: int = 2, 
                       pos_embed_using: bool = True
                       ):
        configs = dict(
            SEQ_LEN = gene_num + 1,
            CLASS = bin_num + 2,
            POS_EMBED_USING = pos_embed_using,
            seed = seed
        )
        self.run_config = configs
                       
    def initialize_model(self) -> None:
        # set seed
        set_seed(self.run_config['seed'])

        self.model = PerformerLM(
            num_tokens = self.run_config['CLASS'],
            dim = 200,
            depth = 6,
            max_seq_len = self.run_config['SEQ_LEN'],
            heads = 10,
            local_attn_heads = 0,
            g2v_position_emb = self.run_config["POS_EMBED_USING"],
            pos_emb_file = os.path.join(self.saved_model_path, 
                                        self.model_files['pos_emb_file'])
            )

    def load_pretrained_model(self) -> None:
        self.initialize_model()

        model_file = os.path.join(self.saved_model_path, 
                                  self.model_files['model_weights'])
        
        msg = f"Loading model from {model_file}"
        log.info(msg)
        try:
            self.model.load_state_dict(torch.load(model_file)["model_state_dict"])
            log.debug(f"Loading all model params from {model_file}")
        except:
            log.warning(f"Loading partial model params from {model_file}")
            # only load params that are in the model and match the size
            model_dict = self.model.state_dict()
            pretrained_dict_full = torch.load(model_file)["model_state_dict"]
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict_full.items()
                if k in model_dict and v.shape  ==  model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                log.debug(f"Loading params {k} with shape {v.shape}")
            
            # print which params are not loaded
            for k, v in model_dict.items():
                if k not in pretrained_dict:
                    log.warning(f"Cannot load {k} with shape {v.shape}")

            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            if torch.cuda.device_count() > 1:
                log.info(f"Using {torch.cuda.device_count()} GPUs")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = torch.nn.DataParallel(self.model)
        
        self.model.to(self.device)

    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        """
        Check if the data is already log1p transformed.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True
    
    def preprocess_data(self, adata_path, layer_key, data_is_raw=True):
        panglao = sc.read_h5ad(os.path.join(self.saved_model_path, 
                                            self.model_files['ref_data']))
        sc.pp.filter_genes(panglao, min_cells=0.05*len(panglao))
        adata = sc.read_h5ad(adata_path)
        counts = sparse.lil_matrix((adata.X.shape[0],panglao.X.shape[1]),dtype=np.float32)
        ref = panglao.var_names.tolist()
        obj = adata.var_names.tolist()

        # copy raw data to adata.X
        if layer_key == "X":
            if data_is_raw and adata.raw is not None:
                adata.X = adata.raw.X.copy()
                del adata.raw
                log.info("Copy raw counts of gene expressions from adata.raw.X")
        else:
            adata.X = adata.layers[layer_key].copy()
            log.info(f"Copy raw counts of gene expressions from adata.layers of {layer_key}")

        for i in range(len(ref)):
            if ref[i] in obj:
                loc = obj.index(ref[i])
                counts[:,i] = adata.X[:,loc]

        counts = counts.tocsr()
        new = ad.AnnData(X=counts)
        new.var_names = ref
        new.obs_names = adata.obs_names
        new.obs = adata.obs
        new.uns = panglao.uns # log1p exist in panglao.uns_keys()

        is_logged = self.check_logged(new)
        if is_logged:
            log.warning(
                "The input data seems to be already log1p transformed. "
                "Set `log1p=False` to avoid double log1p transform."
            )
        sc.pp.filter_cells(new, min_genes=200)
        sc.pp.normalize_total(new, target_sum=1e4)
        sc.pp.log1p(new, base=2)

        return(new)
    
    def prepare_dataloader(
        self,
        data_pt: Dict[str, torch.Tensor],
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Prepare a dataloader from a data_pt

        Args:
            data_pt:                A dictionary with elements such as tokenized 
                                    gene_ids, values, etc.
            batch_size:             Batch size
            per_seq_batch_sample:   If True, sample from each batch of sequences
                                    instead of each sequence
            shuffle:                If True, shuffle the data
            intra_domain_shuffle:   If True, shuffle the data within each batch
            drop_last:              If True, drop the last batch if it is 
                                    smaller than batch_size
            num_workers:            Number of workers for the dataloader; 
                                    if -1, use the number of available CPUs; 
                                    positive integers turn multiprocessing on
        Returns:
            A DataLoader object
        """
        
        if num_workers == -1:
            num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

        dataset = SCDataset(data_pt, CLASS=self.run_config['CLASS'])
    
        data_loader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            drop_last = drop_last,
            num_workers = num_workers,
            pin_memory = True,
        )
        return data_loader

    def get_dataloader(self,
                       adata_path: str,
                       layer_key: str,
                       data_is_raw: bool = False,
                       shuffle: bool = False,
                       drop_last: bool = False) -> None:
        
        data_pt = self.preprocess_data(adata_path, layer_key, data_is_raw)

        msg = "Preparing dataloader"
        log.info(msg)

        self.data_loader = self.prepare_dataloader(
            data_pt.X,
            batch_size = self.batch_size,
            shuffle = shuffle,
            drop_last = drop_last,
            num_workers = self.num_workers,
        )

    def extract_embeddings(self,
                           adata: AnnData,
                           embedding_key: str = "X_scBERT",
                           ) -> Optional[Dict[str, np.ndarray]]:
        
        # check if model is loaded 
        if not self._check_attr("model"):
            msg = "Please load model before extracting embeddings!"
            log.error(msg)
            raise ValueError(msg)
        
        # check if data loader is created
        if not self._check_attr("data_loader"):
            self.get_dataloader()
        
        self.model.eval()

        # save the embeddings to subdir
        embeddings_subdir = os.path.join(self.output_dir, "model_outputs")
        os.makedirs(embeddings_subdir, exist_ok=True)

        # update wandb config
        if self._wandb:
            self._wandb.config.update(self.model_config,
                                      # some config is updated after init
                                      allow_val_change = True)
            self._wandb.config.update(self.run_config,
                                      # some config is updated after init
                                      allow_val_change=True)
        
        msg = "Extracting embeddings"
        log.info(msg)

        cell_embeddings = []

        # how many updates to log
        login_freq = len(self.data_loader) // self.n_log_reports

        for batch, batch_data in enumerate(self.data_loader):
            # process batch data
            full_seq = batch_data.to(self.device)

            if batch % login_freq == 0:
                msg = f"Extracting embeddings for batch {batch+1}/{len(self.data_loader)}"
                log.info(msg)

            with torch.no_grad():
                # Ref: https://github.com/TencentAILabHealthcare/scBERT/issues/70

                x = self.model(full_seq, return_encodings=True)
                # cellemb_cls = x[:, -1, :]
                cellemb_avg = x.mean(dim=1)

            cell_embeddings.append(cellemb_avg.detach().cpu().numpy())
            torch.cuda.empty_cache()

        # flatten the list of cell embeddings
        self.cell_embeddings = np.concatenate(cell_embeddings, axis=0)

        # add embeddings to adata
        adata.obsm[embedding_key] = self.cell_embeddings