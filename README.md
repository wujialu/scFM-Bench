# scFM-Bench

## Introduction
This is the official code repository of the paper **Biology-Driven Insights into the Power of Single-Cell Foundation Models**. Until now, we have included 6 methods as shown below:
| Model | Paper link | Github |
|------------|---------------------------------------------|-------------------------------------------|
| scVI | https://doi.org/10.1038/s41592-018-0229-2 | https://github.com/scverse/scvi-tools |
| Geneformer | https://www.nature.com/articles/s41586-023-06139-9 | https://huggingface.co/ctheodoris/Geneformer |
| scGPT | https://www.nature.com/articles/s41592-024-02201-0 | https://github.com/bowang-lab/scGPT |
| UCE | http://biorxiv.org/lookup/doi/10.1101/2023.11.28.568918 | https://github.com/snap-stanford/UCE |
| scFoundation | https://www.nature.com/articles/s41592-024-02305-7 | https://github.com/biomap-research/scFoundation |
| LangCell | http://arxiv.org/abs/2405.06708 | https://github.com/PharMolix/LangCell |

## Dependencies

Currently the code requires the GPUs supported by flash attention, required for scGPT to run.

GPUs supported by flash attention are:

- Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100).
- Turing GPUs (T4, RTX 2080)

<details>
<summary>Packages version</summary>

This code has been tested with the following versions of the packages:

- Python - tested with `3.8`
- PyTorch - tested with - `1.13.1+cu117`
- CUDA - tested with `11.7`
- lightning - tested with `2.2.0`
- torch-geometric - tested with `2.6.1`
- [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/v1.0.4) - depends on `v1.0.4`
- [scGPT](https://github.com/bowang-lab/scGPT/v0.2.1) - depends on `v0.2.1`
- [Geneformer](https://github.com/jkobject/geneformer/tree/2a9eb7faf7b183ce9f63a5a226321499219a9557) - depends on commit `2a9eb7f`
- [UCE](https://github.com/snap-stanford/UCE/tree/7b31528b84e4c8e7a9717c61e3d03ff7559c61af) - depends on commit `7b31528`
- [scFoundation](https://github.com/biomap-research/scFoundation/tree/948a8ccb950d096148cf03418d870acdcadebd7b) - depends on commit `948a8cc`
- [LangCell](https://github.com/PharMolix/LangCell/tree/a60096bac0321cf3d33046b039af5a0c43fb3df6) - depends on commit `a60096b`
- [scvi-tools](https://github.com/scverse/scvi-tools/tree/1b0ddcf229ca2bc6b0fc44716f2a145f65cd6b8f) - depends on `v0.16.4`
- [sc_foundation_evals](https://github.com/microsoft/zero-shot-scfoundation) - depends on `v0.1.0`
- [scIB](https://github.com/theislab/scib/tree/v1.0.5) - depends on `v1.0.5`
- [Islander](https://github.com/Genentech/Islander/tree/7934aa48b0570e7efb679f0770695eac22c619f0) - depends on commit `7934aa4`

</details>

## Prepare the environment for applying scFMs 
You can download the [conda-packed file](https://zenodo.org/records/14845380/files/singlecell.tar.gz?download=1?), and then unzip it in `${anaconda_install_dir}/envs` (the directory where the anaconda is installed).

```
mkdir ${anaconda_install_dir}/envs/singlecell
tar -xzvf singlecell.tar.gz -C ${anaconda_install_dir}/envs/singlecell
conda activate singlecell
```
We modify the codes in UCE and scFoudnation and apply git patch to sync the modifications.

### UCE
```
git clone https://github.com/snap-stanford/UCE
cd UCE
git checkout 7b31528b84e4c8e7a9717c61e3d03ff7559c61af
git apply ../patch/uce_changes.patch
```
All necessary model files will be downloaded automatically when first running the `eval_single_anndata.py` script.

### scFoundation
```
git clone https://github.com/biomap-research/scFoundation
mv scFoundation xTrimoGene
cd xTrimoGene
git checkout 948a8ccb950d096148cf03418d870acdcadebd7b
git apply ../patch/scfoundation_changes.patch
```

### Others
The scgpt and geneformer packages have been already installed in our provided conda environment.
The codes for applying scGPT, Geneformer and LangCell are located in the `scFM-Bench/sc_foundation_evals` subfolder.


## Download data and checkpoints
Download the datasets and checkpoints of scFMs used in this benchmarking work from [zenodo](https://zenodo.org/records/14795562).
Please unzip `datasets.tar.gz`, `TISCH.tar.gz` and `weights.tar.gz` in the `scFM-Bench/data` directory, which looks like this:
```
├── datasets
│   ├── HLCA_core.h5ad
│   ├── Immune_all_human.h5ad
│   ├── pancreas_scib.h5ad
│   └── Tabula_Sapiens_all.h5ad
├── TISCH
│   ├── Blood
│   │   ├── AEL_GSE142213_CellMetainfo_table.tsv
│   │   ├── AEL_GSE142213_expression.h5
│   │   ├── ALL_GSE132509_CellMetainfo_table.tsv
│   │   ├── ALL_GSE132509_expression.h5
│   │   ├── AML_GSE116256_CellMetainfo_table.tsv
│   │   └── AML_GSE116256_expression.h5
│   ├── Bone
│   │   ├── MM_GSE117156_CellMetainfo_table.tsv
│   │   └── MM_GSE117156_expression.h5
│   ├── Brain
│   │   ├── Glioma_GSE131928_10X_CellMetainfo_table.tsv
│   │   ├── Glioma_GSE131928_10X_expression.h5
│   │   ├── Glioma_GSE138794_CellMetainfo_table.tsv
│   │   ├── Glioma_GSE138794_expression.h5
│   │   ├── Glioma_GSE139448_CellMetainfo_table.tsv
│   │   ├── Glioma_GSE139448_expression.h5
│   │   ├── Glioma_GSE141982_CellMetainfo_table.tsv
│   │   ├── Glioma_GSE141982_expression.h5
│   │   ├── MB_GSE119926_CellMetainfo_table.tsv
│   │   └── MB_GSE119926_expression.h5
│   ├── Eye
│   │   ├── UVM_GSE139829_CellMetainfo_table.tsv
│   │   └── UVM_GSE139829_expression.h5
│   └── preprocess_data.ipynb
└── weights
    ├── Geneformer
    │   ├── default
    │   │   ├── 12L
    │   │   │   ├── config.json
    │   │   │   ├── pytorch_model.bin
    │   │   │   └── training_args.bin
    │   │   └── 6L
    │   │       ├── config.json
    │   │       ├── pytorch_model.bin
    │   │       ├── README.md
    │   │       └── training_args.bin
    │   └── dicts
    │       ├── gene_median_dictionary.pkl
    │       ├── gene_name_id_dict.pkl
    │       └── token_dictionary.pkl
    ├── LangCell
    │   ├── cell_bert
    │   │   ├── config.json
    │   │   └── pytorch_model.bin
    │   ├── cell_proj.bin
    │   ├── config.json
    │   ├── ctm_head.bin
    │   ├── text_bert
    │   │   ├── config.json
    │   │   └── pytorch_model.bin
    │   ├── text_proj.bin
    │   └── tokenizer
    │       └── BiomedBERT
    │           ├── tokenizer_config.json
    │           └── vocab.txt
    ├── scFoundation
    │   └── models.ckpt
    ├── scgpt
    │   └── scGPT_human
    │       ├── args.json
    │       ├── best_model.pt
    │       └── vocab.json
    └── UCE
        └── 33l_8ep_1024t_1280.torch
```
**Note 1**: The TISCH datasets shoule be firstly processed via running the codes in `data/TISCH/preprocess_data.ipynb`.

**Note 2**: The checkpoint for xTrimoGene (scFoundation) should be moved to the `xTrimoGene/model/models` directory.

```
# cd to the scFM-Bench project folder
mv data/weights/scFoundation/models.ckpt xTrimoGene/model/models
```

## Extract pretrained embeddings
### Gene embeddings
```
python 1_extract_gene_embeddings.py
```

### Cell embeddings
**Note**: Please extract geneformer embeddings before LangCell because the data preprocessing is implemented in the geneformer module.
```
# for datasets from scib (Pancreas and Immune)
bash scripts/get_cell_embeddings_scib.sh

# for datasets from cellxgene (HLCA and Tabula Sapiens)
bash scripts/get_cell_embeddings_cellxgene.sh

# for datasets that are already processed (TISCH)
bash scripts/get_cell_embeddings_normalized.sh
```

## Downstream tasks
### Gene function prediction
The baseline code is from [FRoGS](https://github.com/chenhcs/FRoGS). See details in the `scFM-Bench/FRoGS` subfolder.


### Batch integration
The evaluation metrics is based on [scvi-tools](https://github.com/scverse/scvi-tools) and [scGraph](https://github.com/Genentech/Islander).

#### scIB metrics
```
# for datasets from scib
bash scripts/calculate_cluster_metrics.sh

# for datasets from cellxgene
bash scripts/calculate_cluster_metrics_cellxgene.sh
```
Output files: 
- `clustering_metrics.csv`: a csv file contains the results of scIB metrics.
- `X_umap.npy`: a ndarray contains the umap coordinates of cell embeddings.
- `clustering_umap_batch.png`: the umap plot of cell embeddings colored by batch labels.
- `clustering_umap_celltype.png`: the umap plot of cell embeddings colored by cell type labels.

The umap coordinates and png will be saved in the `output` directory.

#### scGraph metrics
```
cd scGraph
python scGraph_cl_ontology.py {dataset_name}
```
By default, the output files will be saved the `output/scGraph` directory. For each specific dataset, there are two output files:
- `{dataset_name}.csv`: a csv file contains the results of the original scGraph and our proposed scGraph-OntoRWR metrics (average across all cell types).
- `{dataset_name}_rwr_detailed.csv`: a csv file contains the cell type-specific scGraph-OntoRWR scores.

### Cell type annotation
The baseline code is from [Onclass](https://github.com/wangshenguiuc/OnClass). See details in the `scFM-Bench/Onclass` subfolder.

### Cancer cell identification
The baseline code is from [SequencingCancerFinder](https://github.com/Patchouli-M/SequencingCancerFinder). See details in the `scFM-Bench/SequencingCancerFinder` subfolder.

### Drug sensitivity prediction
The baseline code is from [SCAD](https://github.com/CompBioT/SCAD). See details in the `scFM-Bench/DrugSensitivity` subfolder.


## Acknowledgments
Our implementation uses microsoft's
[zero-shot-foundation](https://github.com/microsoft/zero-shot-scfoundation) code and 
as a starting point.
Thanks for their great work and code, hope readers of interest could check their work, too.