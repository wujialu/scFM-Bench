# 安装依赖
# pip install cellxgene-census

import cellxgene_census
import pandas as pd

# 打开 2024-07-01 版本 census 数据
version = cellxgene_census.get_census_version_directory()["latest"]['release_build']
# census = cellxgene_census.open_soma(census_version=version)

# 获取 AIDA collection 中的所有 human data
# adata_base = census["census_data"]["homo_sapiens"].ms["RNA"]

donor_ids = ["SG_HEL_H262", "SG_HEL_H269"]
batches = [
    "SG_HEL_B023",
    "SG_HEL_B024",
    "TH_MAH_B001",
    "TH_MAH_B002",
    "TH_MAH_B003",
    "TH_MAH_B004",
    "IN_NIB_B001",
    "IN_NIB_B002"
]
metadata = pd.read_excel("./data/AIDA/AIDA-donor-metadata.xlsx", sheet_name="Table_S1")
metadata = metadata[~metadata["Self-reported ethnicity"].isin(["European"])]
metadata = metadata[metadata["scRNA-seq Experimental Batch"].isin(batches)]
donor_ids += metadata["DCP_ID"].tolist()
print("Total donors to process:", len(donor_ids))

obs_value_filter = f"donor_id in {donor_ids}"
print(obs_value_filter)
with cellxgene_census.open_soma(census_version=version) as census:
    adata = cellxgene_census.get_anndata(
        census = census,
        organism = "Homo sapiens",
        obs_value_filter= obs_value_filter
    )
adata.layers['counts'] = adata.X.copy() 
adata.write_h5ad("data/AIDA/aida_v2_new.h5ad")