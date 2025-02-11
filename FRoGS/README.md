# FRoGS
The source code is from https://github.com/chenhcs/FRoGS.

## Extract pretrained gene embeddings
run the code in `1_extract_gene_embeddings.ipynb` to extract pretrained gene embeddings from five scFMs
the gene embeddings will be saved in the `FRoGS/data` repository.

### Task-specific data
- Tissue specificity prediction: `FRoGS/data/tissue_demo/tissue_specific.txt` (the source data is from the `demo` folder in the FRoGS GitHub repo)
- Gene Ontology prediction: `FRoGS/data/go_terms/Fig.1b.csv` (the source data is from Figure1b in the FRoGS paper)

## Using FRoGS and scFM gene embeddings to classify tissue specific genes 
```
cd demo
python classifier_tissue.py
```
Within the `FRoGS/data/tissue_demo` directory, we have provided three gene lists, each containing tissue-specific genes (Entrez Gene IDs) associated with a specific tissue. In the script `classifier_tissue.py`, we use the FRoGS gene embeddings and scFM gene embeddings to train a MLP-based classifier, and use one-hot embedddings as baseline.

## Using FRoGS and scFM embeddings to predict top-level functions annotated in Gene Ontology
```
cd demo
python classifier_GO.py
```
Within the `FRoGS/data/go_terms` directory, we have provided a gene list with top-level GO annotations (the `Pathway` column) and T-SNE coordinates derived from the FRoGS gene embeddings. In the script `classifier_GO.py`, we use the FRoGS gene embeddings and scFM gene embeddings to train a MLP-based classifier, and use one-hot embedddings as baseline.