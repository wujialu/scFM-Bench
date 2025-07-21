device_id=0
data_folder=/mnt/nvme/extra_data/wujialu/scFM-Bench/data/GRN_analysis
model_folder=/mnt/nvme/extra_data/wujialu/scFM-Bench/data/weights
output_folder=/home/wujialu/scFM-Bench/output

dataset_name=adamson
gene_col=gene_name
save_ext=h5ad
batch_size=1
layer_key=X 
batch_col=str_batch
label_col=celltype # pert condition
data_is_raw=0
pre_normalized=T
normalize_total=1e4
input_type=singlecell
output_type=attention
tgthighres=f1

#! scGPT: assertion error of len_genes
for model_name in scGPT xTrimoGene # Geneformer LangCell scCello 
do
    CUDA_VISIBLE_DEVICES=${device_id} python -u 4_extract_attn_weights.py \
        --data_folder ${data_folder} \
        --model_folder ${model_folder} \
        --output_folder ${output_folder} \
        --dataset_name ${dataset_name} \
        --layer_key ${layer_key} --gene_col ${gene_col} --label_col "${label_col}" \
        --save_ext ${save_ext} \
        --batch_size ${batch_size} \
        --model_name ${model_name} \
        --pre_normalized ${pre_normalized} \
        --data_is_raw ${data_is_raw} --normalize_total ${normalize_total} \
        --batch_col ${batch_col} \
        --input_type ${input_type} --output_type ${output_type} --tgthighres ${tgthighres}
done