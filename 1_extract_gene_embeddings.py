import os 
import sys
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import pickle
from collections import defaultdict
from xTrimoGene.model.load import load_model_frommmf
from transformers import BertForMaskedLM
from sc_foundation_evals import scgpt_forward
from transformers import BertTokenizer, BertModel


gene_set_df = pd.read_csv("./FRoGS/data/gene_id2symbol.csv")

# xTrimoGene
gene_list_file = "./xTrimoGene/OS_scRNA_gene_index.19264.tsv"
gene_list_df = pd.read_csv(gene_list_file, header=0, delimiter='\t')
gene_list = list(gene_list_df['gene_name'])

ckpt_path = "./xTrimoGene/model/models/models.ckpt"
key = "cell"
pretrainmodel, pretrainconfig = load_model_frommmf(ckpt_path, key)
token_emb = pretrainmodel.pos_emb.weight

gene_emb_df = pd.DataFrame(token_emb.detach().numpy()[:19264,:])
gene_emb_df["Symbol"] = gene_list
gene_emb_df = pd.merge(left=gene_emb_df, right=gene_set_df, on="Symbol", how="inner").dropna(inplace=True)
gene_emb_df.set_index("GeneID", inplace=True)
gene_emb_df.drop(["Symbol"], axis=1).to_csv("./FRoGS/data/gene_vec_xtrimogene_768.csv", header=None)

# Geneformer
saved_model_path = "./data/weights/Geneformer/default/12L"
model = BertForMaskedLM.from_pretrained(saved_model_path,
                                        output_attentions=False,
                                        output_hidden_states=True)

dict_paths = "./data/weights/Geneformer/dicts"
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
gene_emb_df = pd.merge(left=gene_set_df, right=gene_emb_df, on="Symbol", how="inner").dropna(inplace=True)
gene_emb_df.set_index("GeneID", inplace=True)
gene_emb_df.drop(["Symbol","ENSG_ID"], axis=1).to_csv("./FRoGS/data/gene_vec_geneformer_512.csv", header=None)

# scGPT
model_dir="./data/weights/scgpt/scGPT_human"
# batch_size depends on available GPU memory; should be a multiple of 8
batch_size=32
# output_dir is the path to which the results should be saved
output_dir="./output/scgpt/scgpt_human/"
# path to where we will store the embeddings and other evaluation outputs
model_out = os.path.join(output_dir, "model_outputs")
# if you can use multithreading specify num_workers
num_workers=0
input_bins=51
model_run="pretrained"
seed=7
n_hvg=1200
# maximum sequence of the input is controlled by max_seq_len, here I'm using the pretrained default
max_seq_len=n_hvg + 1
scgpt_model = scgpt_forward.scGPT_instance(saved_model_path = model_dir,
                                           model_run = model_run,
                                           batch_size = batch_size, 
                                           save_dir = output_dir,
                                           num_workers = num_workers, 
                                           explicit_save_dir = True)
scgpt_model.create_configs(seed = seed, 
                           max_seq_len = max_seq_len, 
                           n_bins = input_bins)
scgpt_model.load_pretrained_model()
token_emb = scgpt_model.model.state_dict()['module.encoder.embedding.weight']
vocab_list = scgpt_model.vocab.get_stoi().keys()
itos = {v: k for k, v in scgpt_model.vocab.get_stoi().items()}
sorted_dict = dict(sorted(itos.items()))

gene_emb_df = pd.DataFrame(token_emb.cpu().numpy())
gene_emb_df["Symbol"] = sorted_dict.values()
gene_emb_df = pd.merge(left=gene_set_df, right=gene_emb_df, on="Symbol", how="left").dropna(inplace=True)
gene_emb_df.set_index("GeneID", inplace=True)
gene_emb_df.drop(["Symbol"], axis=1).to_csv("./FRoGS/data/gene_vec_scgpt_512.csv", header=None)

# LangCell
model = BertModel.from_pretrained('./LangCell/ckpt/cell_bert')
token_emb = model.state_dict()['embeddings.word_embeddings.weight']

gene_emb_df = pd.DataFrame(token_emb.numpy()[:-1, :])
gene_emb_df["ENSG_ID"] = vocab.keys()
gene_emb_df["Symbol"] = gene_emb_df.apply(lambda x: gene_id_name.get(x["ENSG_ID"], None), axis=1)
gene_emb_df = pd.merge(left=gene_set_df, right=gene_emb_df, on="Symbol", how="inner").dropna(inplace=True)
gene_emb_df.set_index("GeneID", inplace=True)
gene_emb_df.drop(["Symbol","ENSG_ID"], axis=1).to_csv("./FRoGS/data/gene_vec_langcell_512.csv", header=None)

# UCE
# NOTE: please run the `UCE/eval_single_anndata.py` script first to download the model files
gene_emb = torch.load("./UCE/model_files/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt")
gene_emb_df = pd.DataFrame(gene_emb).T
gene_emb_df["Symbol"] = gene_emb_df.index
gene_emb_df = pd.merge(left=gene_set_df, right=gene_emb_df, on="Symbol", how="inner").dropna(inplace=True)
gene_emb_df.set_index("GeneID", inplace=True)
gene_emb_df.drop(["Symbol"], axis=1).to_csv("./FRoGS/data/gene_vec_uce_5120.csv", header=None)