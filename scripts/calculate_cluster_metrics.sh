data_folder=/mnt/nvme/extra_data/wujialu/scFM-Bench/data
# output_folder=/home/wujialu/scFM-Bench/output
output_folder=/mnt/nvme/extra_data/wujialu/scFM-Bench/output

# dataset_name=pancreas_scib # sccello
# label_col=celltype

dataset_name=Immune_all_human # sccello
label_col=final_annotation

batch_col=batch
layer_key=counts # for HVG selection

for model in scCello
# for model in Seurat_cca Harmony #HVG scVI Geneformer scGPT UCE xTrimoGene LangCell
do
    python -u 3_cell_clustering.py \
        --data_folder ${data_folder} \
        --output_folder ${output_folder} \
        --model_name ${model} \
        --dataset_name ${dataset_name} \
        --label_col ${label_col} --batch_col ${batch_col} --layer_key ${layer_key} 
done