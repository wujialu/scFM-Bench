device_id=6
# dataset_type=reference
dataset_type=query
data_folder=./data/datasets

# dataset_name=Tabula_Sapiens_all
# label_col=cell_ontology_class_new
# batch_col=tissue_in_publication
# save_ext=h5ad
# ref_dataset_name=HLCA_core
# ref_batch_col=dataset
# scvi_ref_path=./output/${ref_dataset_name}/scVI

dataset_name=HLCA_core
label_col=cell_type
batch_col=dataset
save_ext=loom
ref_dataset_name=Tabula_Sapiens_all
ref_batch_col=tissue_in_publication
scvi_ref_path=./output/${ref_dataset_name}/scVI

# dataset_name=Diabetic_Kidney_Disease
# label_col=cell_type
# batch_col=donor_id  
# save_ext=loom

# data_folder=./data/TISCH
# dataset_name=multi_tissue_tumor_part
# label_col=cell_type
# batch_col=batch # donor_id --> batch
# save_ext=h5ad

layer_key=X
gene_col=feature_name
batch_size=32  
data_is_raw=1
pre_normalized=F
normalize_total=1e4

for model_name in scVI #xTrimoGene scVI scGPT Geneformer LangCell UCE
do
    CUDA_VISIBLE_DEVICES=${device_id} python 2_extract_cell_embeddings.py \
        --data_folder ${data_folder} \
        --dataset_type ${dataset_type} \
        --ref_dataset_name ${ref_dataset_name} --ref_batch_col ${ref_batch_col} \
        --scvi_ref_path ${scvi_ref_path} \
        --dataset_name ${dataset_name} \
        --label_col ${label_col} --batch_col ${batch_col} \
        --layer_key ${layer_key} --gene_col ${gene_col} \
        --save_ext ${save_ext} \
        --batch_size ${batch_size} \
        --model_name ${model_name} \
        --pre_normalized ${pre_normalized} \
        --data_is_raw ${data_is_raw} --normalize_total ${normalize_total} \
        --output_folder './output' 
done