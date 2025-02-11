# Drug Sensitivity Prediction

Source code is from https://github.com/CompBioT/SCAD.

SCAD is used to evaluate all these scFMs in the task of drug sensitivity prediction.

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

