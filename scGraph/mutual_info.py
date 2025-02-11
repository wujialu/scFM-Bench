from __future__ import absolute_import, division, print_function
import sys, time, pandas as pd, numpy as np, scanpy as sc
import Data_Handler as dh
import Utils_Handler as uh
import os
from sklearn.metrics import mutual_info_score

from tqdm import tqdm

"""
Calculate the mutual information scores between the Leiden cluster labels and different metadata label keys
"""

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = "lung_fetal_donor"
        dataset = "skin"
    batch_key = dh.META_[dataset]["batch"]
    label_key = dh.META_[dataset]["celltype"]
    obs_label_keys = dh.META_[dataset]["obs_label_keys"]
    
    # adata = sc.read(dh.DATA_EMB_[dataset], first_column_names=True)
    # bdata = sc.read(dh.DATA_EMB_[dataset + "_hvg"], first_column_names=True)

    # for _obsm in bdata.obsm:
    #     adata.obsm[_obsm + "_hvg"] = bdata.obsm[_obsm]
    
    adata = sc.read(dh.DATA_RAW_[dataset],  first_column_names=True)
    model_list = ["scVI", "Geneformer", "scGPT", "UCE", "xTrimoGene", "LangCell"]
    res_df = pd.DataFrame(columns=obs_label_keys, index=model_list)
    for model in model_list:
        embedding_key = f"X_{model.lower()}"
        if model.lower() == "xtrimogene":
            embedding_file = "mapping_01B-resolution_singlecell_cell_embedding_t4.5_resolution.npy"
        else:
            embedding_file = "cell_emb.npy"
        output_dir = os.path.join(dh.RES_DIR, dataset, model)
        embedding_path = os.path.join(output_dir, embedding_file)
        adata.obsm[embedding_key] = np.load(embedding_path)
        
        sc.pp.neighbors(adata, use_rep=f'X_{model.lower()}')
        sc.tl.leiden(adata, key_added = f'leiden_1_{model.lower()}')
        leiden_labels = adata.obs[f'leiden_1_{model.lower()}']
        
        mutual_info = [mutual_info_score(leiden_labels, adata.obs[key]) for key in obs_label_keys]
        res_df.loc[model, :] = mutual_info

    if not os.path.exists(rf"{dh.RES_DIR}/mutual_info"):
        os.makedirs(rf"{dh.RES_DIR}/mutual_info")
    res_df.to_csv(rf"{dh.RES_DIR}/mutual_info/{dataset}.csv")
    print(res_df)