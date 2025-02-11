# dataset_name=pancreas_scib
# label_col=celltype

dataset_name=Immune_all_human
label_col=final_annotation

batch_col=batch
layer_key=counts # for HVG selection

for model in HVG scVI Geneformer scGPT UCE xTrimoGene LangCell
do
    python 3_cell_clustering.py \
        --model_name ${model} \
        --dataset_name ${dataset_name} \
        --label_col ${label_col} --batch_col ${batch_col} --layer_key ${layer_key} 
done