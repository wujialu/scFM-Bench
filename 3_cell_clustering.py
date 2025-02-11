"""
1. load dataset
2. load pre-computed cell embeddings
3. unsupervised clustering
4. report scib metrics
5. save umap figure (colored by batch_label and celltype_label)
"""

import os
import argparse
from sc_foundation_evals import utils
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_folder', type=str, default='./data/datasets')
    parser.add_argument('--dataset_name', type=str, default='pancreas_scib')
    
    parser.add_argument('--batch_col', type=str, default='batch')
    parser.add_argument('--label_col', type=str, default='celltype')
    parser.add_argument('--layer_key', type=str, default='counts')
    
    parser.add_argument('--model_name', type=str, default='HVG')
    parser.add_argument('--output_folder', type=str, default='./output')
    parser.add_argument('--embedding_file', type=str, default="cell_emb.npy")
    
    args = parser.parse_args()
    return args

def main(args):
    adata = sc.read(args.adata_path)

    #! preprocess data (for selecting HVGs)
    if args.layer_key == "X":
        if adata.raw is not None:
            adata.X = adata.raw.X.copy()
            del adata.raw
            print("Copy raw counts of gene expressions from adata.raw.X")
    else:
        adata.X = adata.layers[args.layer_key].copy()
        print(f"Copy raw counts of gene expressions from adata.layers of {args.layer_key}")
        
    sc.pp.filter_cells(adata, min_genes=25)
    print(adata.X.shape)
    sc.pp.filter_genes(adata, min_cells=10)
    print(adata.X.shape)
    
    #! load cell embeddings
    if args.model_name == "HVG":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor='seurat', subset=False, n_top_genes=2000, batch_key=args.batch_col)
        
        adata.obsm[args.embedding_key] = adata.X[:, adata.var.highly_variable]
        if sparse.issparse(adata.obsm[args.embedding_key]):
            adata.obsm[args.embedding_key] = adata.obsm[args.embedding_key].toarray()
        
    elif args.model_name == "scVI":
        if args.embedding_key not in adata.obsm:
            adata.obsm[args.embedding_key] = np.load(args.embedding_path)
        
    else:
        adata.obsm[args.embedding_key] = np.load(args.embedding_path)
        
    #! calculate scib metrics
    scib_metrics = utils.eval_scib_metrics(adata, 
                                           batch_key=args.batch_col, 
                                           label_key=args.label_col,
                                           embedding_key=args.embedding_key)
    scib_metrics.to_csv(os.path.join(args.output_dir, "clustering_metrics.csv"))
    print("Successfully save scib metrics")
    
    # umap visualization
    umap_file = os.path.join(args.output_dir, "X_umap.npy")
    if os.path.exists(umap_file):
         adata.obsm["X_umap"] = np.load(umap_file)
    else: # generate umap from scratch
        sc.pp.neighbors(adata, use_rep=args.embedding_key)
        sc.tl.umap(adata, min_dist = 0.3)
        np.save(os.path.join(args.output_dir, "X_umap.npy"), adata.obsm["X_umap"])

    sc.set_figure_params(facecolor="white", figsize=(5,4), transparent=True, frameon=False, fontsize=8)
    num_category = adata.obs[args.label_col].nunique()
    ax = sc.pl.umap(adata, color=args.label_col, show=False, 
                    legend_loc=None if num_category > 20 else 'right margin',
                    palette='turbo')
    if args.model_name == "xTrimoGene":
        ax.set_title("scFoundation")
    else:
        ax.set_title(args.model_name)
    plt.savefig(os.path.join(args.output_dir, "clustering_umap_celltype.png"), dpi=300, bbox_inches='tight')
    
    ax = sc.pl.umap(adata, color=args.batch_col, show=False)
    if args.model_name == "xTrimoGene":
        ax.set_title("scFoundation")
    else:
        ax.set_title(args.model_name)
    plt.savefig(os.path.join(args.output_dir, "clustering_umap_batch.png"), dpi=300, bbox_inches='tight')

    #! codes for case study
    # # selected_labels = ['CL:0000786', 'CL:0000980', 'CL:0000787', 'CL:0000788', 'CL:0000823', 'CL:2000055']
    # selected_labels = ['plasma cell',
    #                    'plasmablast',
    #                    'memory b cell',
    #                    'naive b cell',
    #                    'immature natural killer cell',
    #                    'liver dendritic cell']
    # adata.obs['label_col_filtered'] = adata.obs[args.label_col].apply(lambda x: x if x in selected_labels else 'Others')

    # onto_sp = pd.read_csv("/data2/zhuyiheng/wjl/scFoundation/data/OnClass_data_public/Ontology_data/cl.ontology.sp.csv", index_col=0)
    # target_celltype = 'CL:0000786'
    # co2distance = {x: onto_sp.loc[target_celltype, x] for x in adata.obs["cell_type_ontology_term_id"].unique()}
    # adata.obs["DAG distance"] = adata.obs.apply(lambda x: co2distance[x["cell_type_ontology_term_id"]], axis=1)
    # # cmap = sns.cubehelix_palette(adata.obs["DAG distance"].nunique(), rot=-.25, light=.9, as_cmap=True)
    # ax = sc.pl.umap(adata, color="DAG distance", show=False)
    # if args.model_name == "xTrimoGene":
    #     ax.set_title("scFoundation")
    # else:
    #     ax.set_title(args.model_name)
    # plt.tight_layout()
    # plt.savefig(os.path.join(args.output_dir, "clustering_umap_dag_distance.png"), dpi=300, bbox_inches='tight')

    # pal = sns.color_palette("tab10", 6)  
    # palette_dict = {k: pal[i] for i, k in enumerate(selected_labels)}
    # palette_dict["Others"] = 'lightgray'
    # fig = sc.pl.umap(adata, color="label_col_filtered", return_fig=True, 
    #                  palette = palette_dict, alpha=0.6)
    # plt.tight_layout()
    # fig.savefig(os.path.join(args.output_dir, "clustering_umap_selected_labels.png"), dpi=300, bbox_inches='tight')

    print(f"Successfully save UMAP figure for dataset {args.dataset_name} with model {args.model_name}")
    
if __name__ == "__main__":
    args = args_parser()

    args.adata_path = os.path.join(args.data_folder, f"{args.dataset_name}.h5ad")
    args.embedding_key = f"X_{args.model_name.lower()}"
    args.output_dir = os.path.join(args.output_folder, args.dataset_name, args.model_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.model_name.lower() == "xtrimogene":
        args.embedding_file = "mapping_01B-resolution_singlecell_cell_embedding_t4.5_resolution.npy"
    args.embedding_path = os.path.join(args.output_dir, args.embedding_file)
    
    main(args)