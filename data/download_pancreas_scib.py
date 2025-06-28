import scanpy as sc

pancreas_adata_path = "../data/datasets/pancreas_scib.h5ad"

pancreas_adata = sc.read(
    pancreas_adata_path,
    backup_url="https://figshare.com/ndownloader/files/24539828",
)

pancreas_adata.obs['batch'] = (
    pancreas_adata.obs["tech"]
    .str.lower()
    # merge indrop as those are all the same technology  
    .str.replace("indrop[0-9]*", "indrop", regex=True)
    )

pancreas_adata.write_h5ad(pancreas_adata_path, compression="gzip")