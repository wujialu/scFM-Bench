device_id=0
dataset_type=reference
data_folder=/mnt/nvme/extra_data/wujialu/scFM-Bench/data/datasets
model_folder=/mnt/nvme/extra_data/wujialu/scFM-Bench/data/weights
output_folder=/mnt/nvme/extra_data/wujialu/scFM-Bench/output
# output_folder=/home/wujialu/scFM-Bench/output

dataset_name=pancreas_scib
label_col=celltype
# dataset_name=Immune_all_human
# label_col=final_annotation

batch_col=batch
layer_key=counts
gene_col=gene_symbols
save_ext=loom # h5ad may cause issues
batch_size=8  # larger batch_size for scVI
data_is_raw=1
pre_normalized=F
normalize_total=1e4
pass_cell_cls=1
normalize=1

for model_name in scCello #scBERT HVG Harmony scVI UCE xTrimoGene scVI Geneformer scGPT scCello 
do
    CUDA_VISIBLE_DEVICES=${device_id} python 2_extract_cell_embeddings.py \
        --data_folder ${data_folder} \
        --model_folder ${model_folder} \
        --dataset_type ${dataset_type} \
        --dataset_name ${dataset_name} \
        --label_col ${label_col} --batch_col ${batch_col} \
        --layer_key ${layer_key} --gene_col ${gene_col} \
        --save_ext ${save_ext} \
        --pre_normalized ${pre_normalized} \
        --data_is_raw ${data_is_raw} --normalize_total ${normalize_total} \
        --batch_size ${batch_size} \
        --model_name ${model_name} \
        --output_folder ${output_folder} \
        --pass_cell_cls ${pass_cell_cls} \
        --normalize ${normalize}
done
# > logs/$(date +%Y%m%d-%H-%M-%S).log 2>&1