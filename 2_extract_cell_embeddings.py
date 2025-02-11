import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
from loggers import init_logger
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import matplotlib.pyplot as plt
import scvi
from sc_foundation_evals import geneformer_forward, scgpt_forward, langcell_forward
from sc_foundation_evals import data
import subprocess
import time
from functools import wraps

import warnings
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")

import wandb

def calculate_params(model):
    total_params = sum(
	param.numel() for param in model.parameters()
    )
    return total_params

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_folder', type=str, default='./data/datasets')
    parser.add_argument('--dataset_name', type=str, default='pancreas_scib')
    parser.add_argument('--dataset_type', type=str, default='reference')
    parser.add_argument('--ref_dataset_name', type=str, default='Tabula_Sapiens_all')
    parser.add_argument('--scvi_ref_path', type=str, default='./output/Tabula_Sapiens_all/scVI/model.pt')
    parser.add_argument('--ref_batch_col', type=str, default=None)
    
    parser.add_argument('--batch_col', type=str, default=None)
    parser.add_argument('--label_col', type=str, default=None)
    parser.add_argument('--gene_col', type=str, default='gene_symbols')
    
    parser.add_argument('--model_name', type=str, default='scVI')
    parser.add_argument('--output_folder', type=str, default='./output')
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=-1)
    
    parser.add_argument('--data_is_raw', type=int, default=1)
    parser.add_argument('--normalize_total', type=float, default=1e4)
    
    # params for Geneformer
    parser.add_argument('--save_ext', type=str, default='loom', choices=["loom", "h5ad"])
    
    # params for scGPT
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--n_bins', type=int, default=51)
    parser.add_argument('--layer_key', type=str, default='counts')
    
    # params for UCE
    parser.add_argument('--species', type=str, default="human")
    
    # params for xTrimo
    parser.add_argument('--input_type', type=str, default='singlecell',choices=['singlecell','bulk'], help='input type; default: singlecell')
    parser.add_argument('--output_type', type=str, default='cell',choices=['cell','gene','gene_batch','gene_expression'], help='cell or gene embedding; default: cell the difference between gene and gene_batch is that in gene mode the gene embedding will be processed one by one. while in gene_batch mode, the gene embedding will be processed in batch. GEARS use gene_batch mode.')
    parser.add_argument('--pool_type', type=str, default='all',choices=['all','max'], help='pooling type of cell embedding; default: all only valid for output_type=cell')
    parser.add_argument('--tgthighres', type=str, default='t4.5', help='the targeted high resolution (start with t) or the fold change of the high resolution (start with f), or the addtion (start with a) of the high resoultion. only valid for input_type=singlecell')
    parser.add_argument('--pre_normalized', type=str, default='F',choices=['F','T','A'], help='if normalized before input; default: False (F). choice: True(T), Append(A) When input_type=bulk: pre_normalized=T means log10(sum of gene expression). pre_normalized=F means sum of gene expression without normalization. When input_type=singlecell: pre_normalized=T or F means gene expression is already normalized+log1p or not. pre_normalized=A means gene expression is normalized and log1p transformed. the total count is appended to the end of the gene expression matrix.')
    parser.add_argument('--version',  type=str, default='rde', help='only valid for output_type=cell. For read depth enhancemnet, version=rde For others, version=ce')
    
    args = parser.parse_args()
    return args


def monitor_inference_resources(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Inference function '{func.__name__}' executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to select target genes encode protein 
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        gene_list->list: wanted target gene 
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """
    # to_fill = np.setdiff1d(gene_list, adata.var.index.values) # generate to fill list
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                              columns=to_fill_columns, 
                              index=X_df.index)
    #     df = pd.concat([X_df, padding_df], axis=1)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns, var

@monitor_inference_resources
def run_scvi(args):
    adata = sc.read(args.adata_path)
    if args.layer_key == "X":
        if args.data_is_raw and adata.raw is not None:
            adata.X = adata.raw.X.copy()
            del adata.raw
            args.logger.info("Copy raw counts of gene expressions from adata.raw.X")
    else:
        adata.X = adata.layers[args.layer_key].copy()
        args.logger.info(f"Copy raw counts of gene expressions from adata.layers of {args.layer_key}")
    
    sc.pp.filter_cells(adata, min_genes=25) 
    args.logger.info(f"After filter cells: {adata.X.shape}")
    sc.pp.filter_genes(adata, min_cells=10)
    args.logger.info(f"After filter genes {adata.X.shape}")
    if args.data_is_raw:
        sc.pp.normalize_total(adata, target_sum=args.normalize_total)
        sc.pp.log1p(adata)
    
    #! test the performance with (scVI folder) or w/o batch key (scVI_hvg_wo_batchkey folder) for selection
    #! result: no significant difference for scIB, but significant to cell type annotation (Tabula dataset)
    sc.pp.highly_variable_genes(adata, flavor="seurat", subset=False, n_top_genes=args.n_hvg, batch_key=args.batch_col)
    adata = adata[:, adata.var.highly_variable].copy()
    args.logger.info(f"The shape of the processed data: {adata.X.shape}")
    
    scvi.model.SCVI.setup_anndata(adata, batch_key=args.batch_col) #! use raw counts or SoupX-corrected counts as input
    # Create the SCVI model instance with ZINB loss as default. 
    # Insert “gene_likelihood=’nb’,” to change the reconstruction loss to NB loss.
    vae = scvi.model.SCVI(adata, n_layers=2, n_latent=30)
    vae.train(use_gpu=True) #! use the default setting of epochs

    vae.save(args.output_dir, overwrite=True)
    
    cell_embeddings = vae.get_latent_representation()

    args.logger.info(f"The shape of the extracted cell embeddings: {cell_embeddings.shape}")  
    emb_file = f"cell_emb_wo_batch.npy" if args.batch_col is None else f"cell_emb_{args.batch_col}.npy"
    # emb_file = f"cell_emb_wo_batch.npy" if args.batch_col is None else f"cell_emb.npy"
    np.save(os.path.join(args.output_dir, emb_file), cell_embeddings)
    
@monitor_inference_resources
def run_scvi_surgery(args):
    query_adata = sc.read(args.adata_path)

    def process_adata(adata):
        if args.layer_key == "X":
            if args.data_is_raw and adata.raw is not None:
                adata.X = adata.raw.X.copy()
                del adata.raw
                args.logger.info("Copy raw counts of gene expressions from adata.raw.X")
        else:
            adata.X = adata.layers[args.layer_key].copy()
            args.logger.info(f"Copy raw counts of gene expressions from adata.layers of {args.layer_key}")
        
        sc.pp.filter_cells(adata, min_genes=25) 
        args.logger.info(f"After filter cells: {adata.X.shape}")
        sc.pp.filter_genes(adata, min_cells=10)
        args.logger.info(f"After filter genes {adata.X.shape}")
        if args.data_is_raw:
            sc.pp.normalize_total(adata, target_sum=args.normalize_total)
            sc.pp.log1p(adata)
            #!!! WARNING: adata.X seems to be already log-transformed.(adata.uns.log1p exists)
        return adata
    
    query_adata = process_adata(query_adata)
    query_adata.obs[args.ref_batch_col] = query_adata.obs[args.batch_col]
    scvi.model.SCVI.prepare_query_anndata(query_adata, args.scvi_ref_path)
    scvi_query = scvi.model.SCVI.load_query_data(
        query_adata,
        args.scvi_ref_path,
        use_gpu=True
    )
    scvi_query.train(use_gpu=True)
    scvi_query.save(args.output_dir, overwrite=True)
    
    cell_embeddings = scvi_query.get_latent_representation()

    args.logger.info(f"The shape of the extracted cell embeddings: {cell_embeddings.shape}")  
    emb_file = f"cell_emb_wo_batch.npy" if args.batch_col is None else "cell_emb.npy"
    np.save(os.path.join(args.output_dir, emb_file), cell_embeddings)

@monitor_inference_resources
def run_geneformer(args):
    geneform = geneformer_forward.Geneformer_instance(save_dir = args.output_dir, 
                                                      saved_model_path = args.model_dir,
                                                      explicit_save_dir = True,
                                                      num_workers = args.num_workers,
                                                      batch_size=args.batch_size)
    geneform.load_pretrained_model()
    geneform.load_vocab(args.dict_dir)

    total_params = calculate_params(geneform.model)
    print("Total params for the current model: {:.1f} million".format(total_params/1e6))
    
    dataset_name = os.path.basename(args.adata_path).split(".")[0]
    processed_adata_path = os.path.join(args.preprocessed_dir, f"{dataset_name}.{args.save_ext}")
    if not os.path.exists(os.path.join(args.preprocessed_dir, f"{dataset_name}.dataset")):
        input_data = data.InputData(adata_dataset_path = args.adata_path)
        
        if not os.path.exists(processed_adata_path):
            input_data.preprocess_data(gene_col = args.gene_col,
                                       model_type = "geneformer",
                                       save_ext = args.save_ext, 
                                       gene_name_id_dict = geneform.gene_name_id,
                                       preprocessed_path = args.preprocessed_dir,
                                       counts_layer = args.layer_key,
                                       data_is_raw = args.data_is_raw)
        
        geneform.tokenize_data(adata_path = processed_adata_path,
                               dataset_path = args.preprocessed_dir,
                               cell_type_col = args.label_col,
                               data_is_raw = args.data_is_raw)
    else:
        geneform.load_tokenized_dataset(os.path.join(args.preprocessed_dir, f"{args.dataset_name}.dataset"))
        input_data = data.InputData(adata_dataset_path = processed_adata_path)
    
    args.logger.info(f"The shape of the processed data: {input_data.adata.X.shape}")
    geneform.extract_embeddings(data = input_data,
                                batch_size = args.batch_size, 
                                layer = -2) # recommend use the 2nd to last layer 
    
    cell_embeddings = geneform.cell_embeddings
    args.logger.info(f"The shape of the extracted cell embeddings: {cell_embeddings.shape}")  
    np.save(os.path.join(args.output_dir, "cell_emb.npy"), cell_embeddings)
    
@monitor_inference_resources
def run_scgpt(args):
    scgpt_model = scgpt_forward.scGPT_instance(saved_model_path = args.model_dir,
                                               model_run = "pretrained",
                                               batch_size = args.batch_size, 
                                               save_dir = args.output_dir,
                                               num_workers = args.num_workers, 
                                               explicit_save_dir = True)
    scgpt_model.create_configs(seed = args.seed, 
                               max_seq_len = args.n_hvg+1, 
                               n_bins = args.n_bins)
    scgpt_model.load_pretrained_model()
    total_params = calculate_params(scgpt_model.model)
    print("Total params for the current model: {:.1f} million".format(total_params/1e6))
    
    input_data = data.InputData(adata_dataset_path = args.adata_path)
    vocab_list = scgpt_model.vocab.get_stoi().keys()
    #! adata.X is normalized & log_transformed (use adata.raw.X or adata.layers["counts"])
    input_data.preprocess_data(gene_vocab = vocab_list,
                               model_type = "scGPT",
                               gene_col = args.gene_col,
                               data_is_raw = args.data_is_raw, 
                               normalize_total=args.normalize_total if args.data_is_raw else 0,
                               counts_layer = args.layer_key, 
                               n_bins = args.n_bins,
                               n_hvg = args.n_hvg
                               )
    if args.batch_col is not None:
        input_data.add_batch_labels(batch_key = args.batch_col)
    scgpt_model.tokenize_data(data = input_data,
                              input_layer_key = "X_binned",
                              include_zero_genes = False)
    args.logger.info(f"The shape of the processed data: {input_data.adata.X.shape}")
    scgpt_model.extract_embeddings(data = input_data, save_outputs=False)
    
    cell_embeddings = scgpt_model.cell_embeddings
    args.logger.info(f"The shape of the extracted cell embeddings: {cell_embeddings.shape}")  
    np.save(os.path.join(args.output_dir, "cell_emb.npy"), cell_embeddings)
    
@monitor_inference_resources
def run_uce(args):
    os.chdir("./UCE")
    command = ["python3", "eval_single_anndata.py",
               "--adata_path", args.adata_path,
               "--dir", args.output_dir,
               "--species", args.species,
               "--model_loc", args.model_loc,
               "--batch_size", str(args.batch_size), #! cannot pass int
               "--nlayers", "33",
               "--layer_key", args.layer_key,
               "--gene_col", args.gene_col,
               "--data_is_raw", str(args.data_is_raw),
               "--skip", "1"] 
    
    subprocess.run(command)

@monitor_inference_resources
def run_xtrimo(args):
    # to 19264
    adata = sc.read(args.adata_path)
    if args.layer_key == "X":
        if args.data_is_raw and adata.raw is not None:
            adata.X = adata.raw.X.copy()
            del adata.raw
            args.logger.info("Copy raw counts of gene expressions from adata.raw.X")
    else:
        adata.X = adata.layers[args.layer_key].copy()
        args.logger.info(f"Copy raw counts of gene expressions from adata.layers of {args.layer_key}")
        
    if sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix.toarray(adata.X) 
    
    if args.gene_col in adata.var.columns:
        columns = adata.var[args.gene_col].tolist()
    else:
        columns = adata.var.index.tolist()
    X_df= pd.DataFrame(adata.X,index=adata.obs.index.tolist(),columns=columns) # read from csv file
    gene_list_df = pd.read_csv('./xTrimoGene/OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])
    X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list)
    
    preprocessed_dir = os.path.dirname(args.adata_path) + "/xTrimoGene/"
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)
    dataset_name = os.path.basename(args.adata_path).split(".")[0]
    preprocessed_path = os.path.join(preprocessed_dir, f"{dataset_name}_19264_{args.layer_key}.npy")
    np.save(preprocessed_path, X_df.to_numpy())
    
    os.chdir("./xTrimoGene/model")
    command = ["python3", "get_embedding.py",
               "--task_name", "mapping",
               "--input_type", args.input_type,
               "--output_type", args.output_type,
               "--pool_type", args.pool_type,
               "--tgthighres", args.tgthighres,
               "--data_path", preprocessed_path,
               "--save_path", args.output_dir,
               "--pre_normalized", args.pre_normalized,
               "--version", args.version]
    
    subprocess.run(command)

@monitor_inference_resources
def run_langcell(args):
    langcell_model = langcell_forward.Langcell_instance(saved_model_path = args.model_dir,
                                                        saved_tokenizer_path = args.tokenizer_dir,
                                                        batch_size = args.batch_size,
                                                        save_dir = args.output_dir,
                                                        num_workers = args.num_workers, 
                                                        explicit_save_dir = True)
    langcell_model.load_pretrained_model()
    langcell_model.load_tokenizer()
    total_params = calculate_params(langcell_model.model)
    print("Total params for the current model: {:.1f} million".format(total_params/1e6))

    dataset_name = os.path.basename(args.adata_path).split(".")[0]
    langcell_model.load_tokenized_dataset(os.path.join(args.preprocessed_dir, f"{dataset_name}.dataset"))
    langcell_model.get_dataloader()
    
    # sc.read(.h5ad or .loom)
    processed_adata_path = os.path.join(args.preprocessed_dir, f"{dataset_name}.{args.save_ext}")
    input_data = data.InputData(adata_dataset_path = processed_adata_path)
    
    langcell_model.extract_embeddings(data = input_data)
    cell_embeddings = langcell_model.cell_embeddings
    args.logger.info(f"The shape of the extracted cell embeddings: {cell_embeddings.shape}")  
    np.save(os.path.join(args.output_dir, "cell_emb.npy"), cell_embeddings)


def main(args):
    args.logger.info(f"Extract cell embeddeings using {args.model_name}")
    
    if args.model_name.lower() == "scvi":
        args.n_hvg = 2000
        if args.dataset_type == "query":
            args.output_dir = args.output_dir + "_surgery"
            run_scvi_surgery(args)
        else:
            run_scvi(args)
        
    elif args.model_name.lower() == "geneformer":
        args.model_dir = "./data/weights/Geneformer/default/12L"
        args.dict_dir = "./data/weights/Geneformer/dicts"
        args.preprocessed_dir = f"./data/datasets/geneformer/{args.dataset_name}/{args.layer_key}"
        if not os.path.exists(args.preprocessed_dir):
            os.makedirs(args.preprocessed_dir)
        run_geneformer(args)
        
    elif args.model_name.lower() == "scgpt":
        args.model_dir = "./data/weights/scgpt/scGPT_human"
        args.n_hvg = 1200
        run_scgpt(args)
    
    elif args.model_name.lower() == "uce":
        args.model_loc = "./data/weights/UCE/33l_8ep_1024t_1280.torch"
        args.output_dir = args.output_dir + "/"
        run_uce(args)
    
    elif args.model_name.lower() == "xtrimogene":
        run_xtrimo(args)
    
    elif args.model_name.lower() == "langcell":
        args.model_dir = "./data/weights/LangCell"
        args.tokenizer_dir = "./data/weights/LangCell/tokenizer/BiomedBERT"
        args.preprocessed_dir = f"./data/datasets/geneformer/{args.dataset_name}/{args.layer_key}"
        run_langcell(args)
    
    else:
        args.logger.error(f"The model name {args.model_name} is invalid")
    

if __name__ == "__main__":
    args = args_parser()

    args.adata_path = os.path.join(args.data_folder, f"{args.dataset_name}.h5ad")
    args.ref_adata_path = os.path.join(args.data_folder, f"{args.ref_dataset_name}.h5ad")
    args.embedding_key = f"X_{args.model_name.lower()}"
    args.output_dir = os.path.join(args.output_folder, args.dataset_name, args.layer_key, args.model_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    log_file_name = os.path.join(args.output_dir, "log.txt")
    args.logger = init_logger(log_file_name)
    
    # wandb.init(
    #     project="scFoundation",
    #     entity="violet-storm",
    #     name=f"{args.dataset_name}-{args.model_name}",
    #     config={
    #         "model": args.model_name
    #     }
    # )

    main(args)