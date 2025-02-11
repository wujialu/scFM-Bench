# 	Cancer-Finder
Paper: Domain generalization enables general cancer cell annotation in single-cell and spatial transcriptomics  

The source code is from https://github.com/Patchouli-M/SequencingCancerFinder.

### Prepare data
Download and put the TISCH data in the `scFM-Bench/data/TISCH` folder.

#### Extract scVI embeddings
Run the code in `scFM-Bench/data/TISCH/preprocess_data.ipynb`, combine all datasets into a single dataset `TISCH_combined.h5ad`.

Since that the gene expression profile downloaded from the TISCH database is already processed by normalization and log1p transformation, please run the command:
```
cd ../
bash scripts/get_cell_embeddings_normalized.sh
```

The extracted cell embeddings will be saved in the `scFM-Bench/output/TISCH_combined` folder.

#### Extract scFM embeddings
Directly use scFMs to extract cell embeddings for each dataset.

Since that the gene expression profile downloaded from the TISCH database is already processed by normalization and log1p transformation, please run the command:
```
cd ../
bash scripts/get_cell_embeddings_normalized.sh
```

The extracted cell embeddings will be saved in the `scFM-Bench/output/TISCH` folder.

#### Gather cell embeddings into tissue-specific datasets
Run the code in `scFM-Bench/data/TISCH/preprocess_data.ipynb`

The tissue-specific h5ad files will be saved in the `scFM-Bench/output/TISCH` folder.

### Train the classification model
If you want train the Cancer-Finder model with Optuna-based hyperparameter optimization, please run the command:
```
cd SequencingCancerFinder
bash train_opt.sh
```

If you want train the Cancer-Finder model with pre-defined hyperparameters, please run the command:
```
cd SequencingCancerFinder
bash train.sh
```

Please specify the `input_features` in `train_opt.sh` and `train.sh` for training embedding-based model.
