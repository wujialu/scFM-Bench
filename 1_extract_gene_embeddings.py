import os 
import sys
sys.path.insert(0, "./sc_foundation_evals")
sys.path.insert(0, "./xTrimoGene/model")
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import pickle
from collections import defaultdict
from xTrimoGene.model.load import load_model_frommmf
from sc_foundation_evals import scgpt_forward
from transformers import BertForMaskedLM, BertModel
from sc_foundation_evals.sccello.src.model_prototype_contrastive import PrototypeContrastiveModel


gene_set_df = pd.read_csv("./FRoGS/data/gene_id2symbol.csv")
parent_model_dir = "/mnt/nvme/extra_data/wujialu/scFM-Bench/data/weights"

# ==================== xTrimoGene ====================
gene_list_file = f"{parent_model_dir}/scFoundation/OS_scRNA_gene_index.19264.tsv"
gene_list_df = pd.read_csv(gene_list_file, header=0, delimiter='\t')
gene_list = list(gene_list_df['gene_name'])

ckpt_path = f"{parent_model_dir}/scFoundation/models.ckpt"
key = "cell"
pretrainmodel, pretrainconfig = load_model_frommmf(ckpt_path, key)
token_emb = pretrainmodel.pos_emb.weight

gene_emb_df = pd.DataFrame(token_emb.detach().numpy()[:19264,:])
gene_emb_df["Symbol"] = gene_list
gene_emb_df = pd.merge(left=gene_set_df, right=gene_emb_df, on="Symbol", how="inner")
gene_emb_df.set_index("GeneID", inplace=True)
gene_emb_df.drop(["Symbol"], axis=1, inplace=True)
gene_emb_df.to_csv(f"./FRoGS/gene_embs/gene_vec_xtrimogene_{gene_emb_df.shape[1]}.csv", header=None)

print("Gene embeddings extracted from xTrimoGene saved")

# ==================== Geneformer ===================
saved_model_path = f"{parent_model_dir}/Geneformer/default/12L"
model = BertForMaskedLM.from_pretrained(saved_model_path,
                                        output_attentions=False,
                                        output_hidden_states=True)

dict_paths = f"{parent_model_dir}/Geneformer/dicts"
token_dictionary_path = os.path.join(dict_paths, "token_dictionary.pkl")
with open(token_dictionary_path, "rb") as f:
    vocab = pickle.load(f)

pad_token_id = vocab.get("<pad>")

gene_name_id_path = os.path.join(dict_paths, "gene_name_id_dict.pkl")
with open(gene_name_id_path, "rb") as f:
    gene_name_id = pickle.load(f)

token_emb = model.state_dict()['bert.embeddings.word_embeddings.weight']
gene_id_name = {v: k for k, v in gene_name_id.items()}

gene_emb_df = pd.DataFrame(token_emb.numpy())
gene_emb_df["ENSG_ID"] = vocab.keys()
gene_emb_df["Symbol"] = gene_emb_df.apply(lambda x: gene_id_name.get(x["ENSG_ID"], None), axis=1)
gene_emb_df = pd.merge(left=gene_set_df, right=gene_emb_df, on="Symbol", how="inner")
gene_emb_df.set_index("GeneID", inplace=True)
gene_emb_df.drop(["Symbol","ENSG_ID"], axis=1, inplace=True)
gene_emb_df.to_csv(f"./FRoGS/gene_embs/gene_vec_geneformer_{gene_emb_df.shape[1]}.csv", header=None)

print("Gene embeddings extracted from Geneformer saved")

# ==================== LangCell ===================
model = BertModel.from_pretrained(f"{parent_model_dir}/LangCell/cell_bert")
token_emb = model.state_dict()['embeddings.word_embeddings.weight']

gene_emb_df = pd.DataFrame(token_emb.numpy()[:-1, :]) # remove cls token
gene_emb_df["ENSG_ID"] = vocab.keys()
gene_emb_df["Symbol"] = gene_emb_df.apply(lambda x: gene_id_name.get(x["ENSG_ID"], None), axis=1)
gene_emb_df = pd.merge(left=gene_set_df, right=gene_emb_df, on="Symbol", how="inner")
gene_emb_df.set_index("GeneID", inplace=True)
gene_emb_df.drop(["Symbol","ENSG_ID"], axis=1, inplace=True)
gene_emb_df.to_csv(f"./FRoGS/gene_embs/gene_vec_langcell_{gene_emb_df.shape[1]}.csv", header=None)

print("Gene embeddings extracted from LangCell saved")

# ==================== scCello ===================
saved_model_path = f"{parent_model_dir}/scCello"
model = PrototypeContrastiveModel.from_pretrained(saved_model_path)
token_emb = model.state_dict()['embeddings.word_embeddings.weight'] # [25427, 256]

gene_emb_df = pd.DataFrame(token_emb.numpy()[:-1, :]) # remove cls token
gene_emb_df["ENSG_ID"] = vocab.keys()
gene_emb_df["Symbol"] = gene_emb_df.apply(lambda x: gene_id_name.get(x["ENSG_ID"], None), axis=1)
gene_emb_df = pd.merge(left=gene_set_df, right=gene_emb_df, on="Symbol", how="inner")
gene_emb_df.set_index("GeneID", inplace=True)
gene_emb_df.drop(["Symbol","ENSG_ID"], axis=1, inplace=True)
gene_emb_df.to_csv(f"./FRoGS/gene_embs/gene_vec_sccello_{gene_emb_df.shape[1]}.csv", header=None)

print("Gene embeddings extracted from scCello saved")

# ==================== scGPT ===================
model_dir=f"{parent_model_dir}/scgpt/scGPT_human"
input_bins=51
seed=7
n_hvg=1200
# maximum sequence of the input is controlled by max_seq_len, here I'm using the pretrained default
max_seq_len=n_hvg + 1
scgpt_model = scgpt_forward.scGPT_instance(saved_model_path = model_dir)
scgpt_model.create_configs(seed = seed, 
                           max_seq_len = max_seq_len, 
                           n_bins = input_bins)
scgpt_model.load_pretrained_model()
# token_emb = scgpt_model.model.state_dict()['module.encoder.embedding.weight']
token_emb = scgpt_model.model.state_dict()['encoder.embedding.weight']
vocab_list = scgpt_model.vocab.get_stoi().keys()
itos = {v: k for k, v in scgpt_model.vocab.get_stoi().items()}
sorted_dict = dict(sorted(itos.items()))

gene_emb_df = pd.DataFrame(token_emb.cpu().numpy())
gene_emb_df["Symbol"] = sorted_dict.values()
gene_emb_df = pd.merge(left=gene_set_df, right=gene_emb_df, on="Symbol", how="inner")
gene_emb_df.set_index("GeneID", inplace=True)
gene_emb_df.drop(["Symbol"], axis=1, inplace=True)
gene_emb_df.to_csv(f"./FRoGS/gene_embs/gene_vec_scgpt_{gene_emb_df.shape[1]}.csv", header=None)

print("Gene embeddings extracted from scGPT saved")

# ==================== UCE ===================
# NOTE: please run the `UCE/eval_single_anndata.py` script first to download the model files
gene_emb = torch.load("./UCE/model_files/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt")
gene_emb_df = pd.DataFrame(gene_emb).T
gene_emb_df["Symbol"] = gene_emb_df.index
gene_emb_df = pd.merge(left=gene_set_df, right=gene_emb_df, on="Symbol", how="inner")
gene_emb_df.set_index("GeneID", inplace=True)
gene_emb_df.drop(["Symbol"], axis=1, inplace=True)
gene_emb_df.to_csv(f"./FRoGS/gene_embs/gene_vec_uce_{gene_emb_df.shape[1]}.csv", header=None)

print("Gene embeddings extracted from UCE saved")

