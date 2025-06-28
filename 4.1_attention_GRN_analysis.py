import copy
import json
import os
from pathlib import Path
import sys
import warnings
import pickle

import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import tqdm
import gseapy as gp
from gears import PertData, GEARS

from scipy.sparse import issparse
import scipy as sp
from einops import rearrange
from torch.nn.functional import softmax
from tqdm import tqdm
import pandas as pd

from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

sys.path.insert(0, "../")

import scgpt as scg
from scgpt.tasks import GeneEmbedding
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.utils import set_seed 
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.preprocess import Preprocessor

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

TF_name = 'BHLHE40'
model_ls = ["scGPT", "Geneformer", "LangCell", "xTrimoGene"]

data_dir = Path("./data/GRN_analysis")
pert_data = PertData(data_dir)
pert_data.load(data_name="adamson")
adata = sc.read(data_dir / "adamson/perturb_processed.h5ad")

adata = adata[adata.obs.condition.isin(['{}+ctrl'.format(TF_name), 'ctrl'])].copy()
groups = adata.obs.groupby('condition').groups

def get_topk_most_influenced_genes(topk, setting):
    attn_top_gene_dict = {}
    attn_top_scores_dict = {}
    for i in groups.keys():
        if i != 'ctrl':
            knockout_gene = i.split('+')[0]
            # knockout_gene_idx = np.where(gene_vocab_idx==vocab([knockout_gene])[0])[0][0]
            knockout_gene_idx = np.where(np.array(dict_sum_condition_mean["gene_names"])==TF_name)[0][0]
            control = dict_sum_condition_mean['ctrl'][:, knockout_gene_idx]
            exp = dict_sum_condition_mean[i][:, knockout_gene_idx]
            # Chnage this line to exp, control, exp-control for three different settings
            if setting == 'difference':
                a = exp-control
            elif setting == 'control':
                a = control
            elif setting == 'perturbed':
                a = exp
            diff_idx = np.argpartition(a, -topk)[-topk:]
            scores = (a)[diff_idx]
            # attn_top_genes = vocab.lookup_tokens(gene_vocab_idx[diff_idx]) + [TF_name]
            attn_top_genes = list(np.array(dict_sum_condition_mean["gene_names"])[diff_idx]) + [TF_name]
            attn_top_gene_dict[i] = list(attn_top_genes)
            attn_top_scores_dict[i] = list(scores)
    return attn_top_gene_dict, attn_top_scores_dict

def score_overlap_genes_by_rank(df, gene_list):
    target_genes = list(df['Target_genes'].values)
    total_targets = len(target_genes)
    gene_to_rank = {gene: rank for rank, gene in enumerate(target_genes)}  # 记录每个基因的排名

    overlap_genes = set(gene_list).intersection(set(target_genes))
    scores = {}

    for gene in overlap_genes:
        rank = gene_to_rank[gene]
        percentile = rank / total_targets
        if percentile < 0.2:
            scores[gene] = 5
        elif percentile < 0.4:
            scores[gene] = 4
        elif percentile < 0.6:
            scores[gene] = 3
        elif percentile < 0.8:
            scores[gene] = 2
        else:
            scores[gene] = 1

    return scores, sum(scores.values()), len(overlap_genes)

for model_name in model_ls:
    print(f"========== Start analysis for {model_name} ==========")
    with open(f'./output/adamson/X/{model_name}/pretrain_attn_condition_mean.pkl', 'rb') as f:
        dict_sum_condition_mean = pickle.load(f)
    
    setting = 'difference' # "control", "perturbed"
    assert setting in ["difference", "control", "perturbed"]
    attn_top_gene_dict_20, attn_top_scores_dict_20 = get_topk_most_influenced_genes(20, setting)
    print(attn_top_scores_dict_20[TF_name + '+ctrl'])
    print(attn_top_gene_dict_20[TF_name + '+ctrl'])

    attn_top_gene_dict_100, attn_top_scores_dict_100 = get_topk_most_influenced_genes(100, setting)
    print(attn_top_scores_dict_100[TF_name + '+ctrl'])
    print(attn_top_gene_dict_100[TF_name + '+ctrl'])

    if setting == 'difference':
        for i in attn_top_gene_dict_20.keys():
            example_genes = attn_top_gene_dict_20[i]
            # gene_idx = [np.where(gene_vocab_idx==vocab([g])[0])[0][0] for g in example_genes]
            gene_idx = [np.where(np.array(dict_sum_condition_mean["gene_names"])==g)[0][0] for g in example_genes]
            scores = dict_sum_condition_mean[i][gene_idx, :][:, gene_idx]-dict_sum_condition_mean['ctrl'][gene_idx, :][:, gene_idx]
            df_scores = pd.DataFrame(data = scores, columns = example_genes, index = example_genes)
            plt.figure(figsize=(6, 6), dpi=300)
            ax = sns.clustermap(df_scores, annot=False, cmap=sns.diverging_palette(145, 300, s=60, as_cmap=True), fmt='.2f', vmin=-0.3, vmax=0.3) 
            plt.savefig(f'./output/adamson/X/{model_name}/heatmap_diff_top20_genes.png', dpi=300)
    

    # validate against CHIP-atlas
    df = pd.read_csv('./data/GRN_analysis/BHLHE40.10.tsv', delimiter='\t')
    gene_list = attn_top_gene_dict_20[TF_name + '+ctrl'][:-1]
    scores, total_score, num_overlap_genes = score_overlap_genes_by_rank(df, gene_list)
    print("Individual scores:", scores)
    print("Total score:", total_score) # max score is 5 * 20 = 100
    print("Number of overlapping genes:", num_overlap_genes)

    # Validate against the Reactome database
    df_database = pd.DataFrame(
        data = [['GO_Biological_Process_2021', 6036],
        ['GO_Molecular_Function_2021', 1274],
        ['Reactome_2022', 1818]],
        columns = ['dataset', 'term'])
    
    databases = ['Reactome_2022']
    m = df_database[df_database['dataset'].isin(databases)]['term'].sum()
    p_thresh = 0.05/((len(groups.keys())-1)*m) # 0.05/1818

    gene_list = attn_top_gene_dict_100[TF_name + '+ctrl']
    enr_Reactome = gp.enrichr(gene_list=gene_list,
                              gene_sets=databases,
                              organism='Human',
                              outdir=f"./output/adamson/X/{model_name}/enrichr_Reactome_{TF_name}",
                              cutoff=0.5) # cutoff for plotting
    out = enr_Reactome.results
    out['Gene List'] = str(gene_list)
    out.to_csv(f"./output/adamson/X/{model_name}/enrichr_Reactome_{TF_name}/enrichr_results.csv")
    print("Enrichment results for Reactome:", len(out))
    out = out[out['P-value'] < p_thresh]
    print("Significant enrichment results for Reactome:", len(out))

# python 4.1_attention_GRN_analysis.py > output/adamson/X/attention_GRN_analysis.log 2>&1