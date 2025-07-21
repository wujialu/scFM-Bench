# For cancer cell identification
device_id=0
data_folder=/mnt/nvme/extra_data/wujialu/scFM-Bench/data/TISCH
model_folder=/mnt/nvme/extra_data/wujialu/scFM-Bench/data/weights
output_folder=/mnt/nvme/extra_data/wujialu/scFM-Bench/output/TISCH

dataset_type=reference
dataset_name=Blood/AEL_GSE142213
gene_col=gene_symbols
label_col="Celltype (malignancy)"
batch_col=dataset
save_ext=h5ad
batch_size=32
layer_key=X 
data_is_raw=0
pre_normalized=T
normalize_total=1e4

# for dataset_name in TISCH_combined
# do
#     for model_name in scVI
#     do
#         for batch_col in Patient dataset tumor tissue
#         do
#             CUDA_VISIBLE_DEVICES=${device_id} python 2_extract_cell_embeddings.py \
#                 --dataset_type ${dataset_type} \
#                 --data_folder ${data_folder} \
#                 --dataset_name ${dataset_name} \
#                 --layer_key ${layer_key} --gene_col ${gene_col} --label_col "${label_col}" \
#                 --save_ext ${save_ext} \
#                 --batch_size ${batch_size} \
#                 --model_name ${model_name} \
#                 --pre_normalized ${pre_normalized} \
#                 --data_is_raw ${data_is_raw} --normalize_total ${normalize_total} \
#                 --batch_col ${batch_col} 
#         done
#     done
# done

model_name=scCello
for tissue in Blood Bone Brain Eye
do
    tissue_data_folder=${data_folder}/${tissue}
    tissue_output_folder=${output_folder}/${tissue}
    for dataset_file in $(ls ${tissue_data_folder}/*.h5ad)
    do
        dataset_name=$(basename ${dataset_file} .h5ad)
        CUDA_VISIBLE_DEVICES=${device_id} python -u 2_extract_cell_embeddings.py \
            --dataset_type ${dataset_type} \
            --data_folder ${tissue_data_folder} \
            --model_folder ${model_folder} \
            --output_folder ${tissue_output_folder} \
            --dataset_name ${dataset_name} \
            --layer_key ${layer_key} --gene_col ${gene_col} --label_col "${label_col}" \
            --save_ext ${save_ext} \
            --batch_size ${batch_size} \
            --model_name ${model_name} \
            --pre_normalized ${pre_normalized} \
            --data_is_raw ${data_is_raw} --normalize_total ${normalize_total}
    done
done