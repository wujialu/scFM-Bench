# Drug Sensitivity Prediction

Source code is from https://github.com/CompBioT/SCAD.

SCAD is used to evaluate all these scFMs in the task of drug sensitivity prediction.

## Data Download
### Processed data
The processed data used in this section can be downloaded from [figrshare](https://figshare.com/articles/dataset/scFoundation_Large_Scale_Foundation_Model_on_Single-cell_Transcriptomics_-_processed_datasets/24049200/3).
Please download the processed data and put it in the `./data/split_norm/` folder.

### Original gene expression data
- `gdsc_rma_raw.csv` was downloaded from https://ibm.ent.box.com/v/paccmann-pytoda-data/folder/91948853171;
plesse rename 'gdsc_rma_raw.csv' as 'gdsc-rma_gene-expression.csv' and put into `./data/original/` folder.
- The scRNA-seq data can be downloaded fully at https://drive.google.com/file/d/15smmSqzVf-6et1EAbPj5LWBdLQ1QkVO9/view?usp=share_link. Please unzip the rar file and get five tsv files. Put them into the `./data/original/gene_count_matrix` folder. For convinience, `scrna_ccle_combined.tsv` is a combination of CPM (count per million) for `scrna_ccle_jhu006_exprs.tsv` and `scrna_ccle_scc47_expr.tsv`.



## Data Preparation

In accordance with our unified cell embedding extraction method, some data format transformation is needed.

```bash
python ./util/csv_to_h5ad.py 
```

SCAD needs bulk and single-cell embeddings to train the model. Please directly use scFMs to extract these embeddings.    

The commands below use *Geneformer* as example.   

```bash
cd ../
mkdir -p ./data/Sorafenib/geneformer/
python 2_extract_cell_embeddings.py --model_name geneformer --drug Sorafenib --data_folder ./data/split_norm/Sorafenib/ --dataset_name Source_Sorafenib --output_folder ./data/Sorafenib/geneformer/
```

Then, split the data using `split_data_SCAD_5fold_norm.py` to conduct 5-fold cross-validation.   

In this repo, `split_data_SCAD_5fold_norm.py` is partly modified to facilitate model/drug selection.

```bash
cd ./data/split_norm/
## without embedding
python split_data_SCAD_5fold_norm.py --drug Sorafenib --emb 0 --software geneformer
## with embedding
python split_data_SCAD_5fold_norm.py --drug Sorafenib --emb 1 --software geneformer
```
## Training SCAD Model

Training SCAD model by using this command:

```bash
python util/SCAD_train_binarized_5folds-pub.py -e FX -d NVP-TAE684 -g _norm -s 42 -h_dim 1024 -z_dim 128 -ep 10 -la1 2 -mbS 8 -mbT 8 -emb 0 --software geneformer
```
Several parameters may not be in the optimal value, so a grid search is needed to find the best hyperparameters.   

Code using Python package Optuna to search optimal hyperparameters is provided in `util/hyperparameter_selection.py`.   

In accordance with the SCAD model paper, baseline and xTrimoGene hyperparameters are not optimized.

```bash
python util/hyperparameter_selection.py
```
By executing this command, several files containing the best hyperparameters and corresponding AUC will be generated in `output/geneformer/best_scad_params_geneformer.txt`.

## Comparing with Baseline
Matplotlib and Jupyter Notebook are used to demonstrate AUCs of models trained *with and without* the scFM-extracted embeddings.   

Jupyter Notebook outputs are saved in `output/jupyter`.   

