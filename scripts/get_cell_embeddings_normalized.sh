device_id=5
dataset_type=reference
data_folder=./data/TISCH
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

for dataset_name in TISCH_combined
do
    for model_name in scVI
    do
        for batch_col in Patient dataset tumor tissue
        do
            CUDA_VISIBLE_DEVICES=${device_id} python 2_extract_cell_embeddings.py \
                --dataset_type ${dataset_type} \
                --data_folder ${data_folder} \
                --dataset_name ${dataset_name} \
                --layer_key ${layer_key} --gene_col ${gene_col} --label_col "${label_col}" \
                --save_ext ${save_ext} \
                --batch_size ${batch_size} \
                --model_name ${model_name} \
                --pre_normalized ${pre_normalized} \
                --data_is_raw ${data_is_raw} --normalize_total ${normalize_total} \
                --batch_col ${batch_col} 
        done
    done
done

# for dataset_name in Blood/AEL_GSE142213 Blood/ALL_GSE132509 Blood/AML_GSE116256 
# for dataset_name in Bone/MM_GSE117156 Eye/UVM_GSE139829 Brain/Glioma_GSE131928_10X 
# for dataset_name in Brain/Glioma_GSE138794 Brain/Glioma_GSE139448 Brain/Glioma_GSE141982 Brain/MB_GSE119926
# do
#     for model_name in UCE
#     do
#         CUDA_VISIBLE_DEVICES=${device_id} python get_cell_embeddings.py \
#             --dataset_type ${dataset_type} \
#             --data_folder ${data_folder} \
#             --dataset_name ${dataset_name} \
#             --layer_key ${layer_key} --gene_col ${gene_col} --label_col "${label_col}" \
#             --save_ext ${save_ext} \
#             --batch_size ${batch_size} \
#             --model_name ${model_name} \
#             --pre_normalized ${pre_normalized} \
#             --data_is_raw ${data_is_raw} --normalize_total ${normalize_total}
#     done
# done