import os

import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix

def number_to_name(drug):
    # 读取数据
    symbol_data = pd.read_csv(
        '../data/processing/HGNC_symbol_all_genes.tsv', sep='\t')
    # 获取 NCBI GENE ID 和 Approved symbol 列，并将 NCBI GENE ID 转换为字符串
    ncbi_gene_ids = symbol_data['NCBI Gene ID'].astype(str).tolist()
    approved_symbols = symbol_data['Approved symbol'].tolist()

    # 创建一个字典用于快速查找
    gene_id_to_symbol = dict(zip(ncbi_gene_ids, approved_symbols))
    data_s = pd.read_csv(
        f'../data/split_norm/{drug}/Source_exprs_resp_z.{drug}.tsv',
        sep='\t', index_col=0)
    # 遍�� data_r 的列名
    new_columns_s = []
    for col in data_s.columns:
        col_str = str(col) + ".0"  # 将列名转换为字符串
        if col_str in gene_id_to_symbol:
            new_columns_s.append(gene_id_to_symbol[col_str])
        else:
            new_columns_s.append(col)
    data_s.columns = new_columns_s

    data_s.to_csv(f'../data/split_norm/{drug}/Source_{drug}_normalised.tsv',
                  sep='\t')

    data_t = pd.read_csv(
        f'../data/split_norm/{drug}/Target_expr_resp_z.{drug}.tsv',
        sep='\t', index_col=0)
    new_columns_t = []
    for col in data_t.columns:
        col_str = str(col) + ".0"
        if col_str in gene_id_to_symbol:
            new_columns_t.append(gene_id_to_symbol[col_str])
        else:
            new_columns_t.append(col)

    data_t.columns = new_columns_t

    data_t.to_csv(f'../data/split_norm/{drug}/Target_{drug}_normalised.tsv', sep='\t')

def normalized_to_original(drug):
    data_s = pd.read_csv(f'../data/split_norm/{drug}/Source_{drug}_normalised.tsv',
                         sep='\t', index_col=0)

    data_t = pd.read_csv(f'../data/split_norm/{drug}/Target_{drug}_normalised.tsv', sep='\t', index_col=0)
    all_s = pd.read_csv("../data/original/gdsc-rma_gene-expression.csv", index_col=0)
    if drug == "PLX4720":
        all_t =pd.read_csv("../data/original/gene_count_matrix/GSE108383_exprs.PLX4720_A375.tsv", sep='\t', index_col=0)
    elif drug == "Etoposide":
        all_t = pd.read_csv("../data/original/gene_count_matrix/GSE149215_exprs.Etoposide_PC9.tsv", sep='\t', index_col=0)
    elif drug =="PLX4720_451Lu":
        all_t = pd.read_csv("../data/original/gene_count_matrix/GSE108383_exprs_PLX4720_451Lu.tsv", sep='\t', index_col=0)
    else:
        all_t = pd.read_csv("../data/original/gene_count_matrix/scrna_ccle_combined.tsv", sep='\t', index_col=0)

    # 遍历 data_s 的每一行
    for idx in data_s.index:
        if idx in all_s.index:
            for col in data_s.columns:
                if col in all_s.columns:
                    # 用 all 中的数据替换 data_s 中的原有数据
                    data_s.at[idx, col] = all_s.at[idx, col]

    for idx in data_t.index:
        if idx in all_t.index:
            for col in data_t.columns:
                if col in all_t.columns:
                    # 用 all 中的数据替换 data_s 中的原有数据
                    data_t.at[idx, col] = all_t.at[idx, col]

    # 保存替换后的 data_s
    data_s.to_csv(
        f'../data/split_norm/{drug}/Source_{drug}.tsv',
        sep='\t')

    data_t.to_csv(f"../data/split_norm/{drug}/Target_{drug}.tsv", sep='\t')


def source_csv_to_h5ad(drug):
    # Load the CSV file
    df = pd.read_csv(f'../data/split_norm/{drug}/Source_{drug}.tsv', sep="\t", index_col=0, usecols=lambda col: col not in ['logIC50', 'response'])

    # Convert the DataFrame to a sparse matrix
    sparse_matrix = csr_matrix(df.values)

    # Convert the index to a DataFrame for 'obs'
    obs_df = pd.DataFrame(index=df.index)

    # Create an AnnData object with the sparse matrix as X and the index DataFrame as obs
    adata = ad.AnnData(X=sparse_matrix, obs=obs_df, var=pd.DataFrame(index=df.columns))
    adata.layers['counts'] = adata.X.copy()
    
    # Save the AnnData object
    os.mkdir('../data/datasets') if not os.path.exists('../data/datasets') else None
    adata.write(f'../data/datasets/Source_{drug}.h5ad')

def target_csv_to_h5ad(drug):
    # Load the CSV file
    df = pd.read_csv(f'../data/split_norm/{drug}/Target_{drug}.tsv', sep="\t", index_col=0, usecols=lambda col: col not in ['response'])

    # Convert the DataFrame to a sparse matrix
    sparse_matrix = csr_matrix(df.values)

    # Convert the index to a DataFrame for 'obs'
    obs_df = pd.DataFrame(index=df.index)

    # Create an AnnData object with the sparse matrix as X and the index DataFrame as obs
    adata = ad.AnnData(X=sparse_matrix, obs=obs_df, var=pd.DataFrame(index=df.columns))
    adata.layers['counts'] = adata.X.copy()
    # Save the AnnData object
    os.mkdir('../data/datasets') if not os.path.exists('../data/datasets') else None
    adata.write(f'../data/datasets/Target_{drug}.h5ad')

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug", type=str, help="Drug name", choices=["NVP-TAE684", "Sorafenib", "Etoposide", "PLX4720_451Lu"])
    args = parser.parse_args()
    return args

def main(args):
    drug = args.drug
    number_to_name(drug)
    normalized_to_original(drug)
    source_csv_to_h5ad(drug)
    target_csv_to_h5ad(drug)