data_folder=/mnt/nvme/extra_data/wujialu/scFM-Bench/data/datasets
output_folder=/mnt/nvme/extra_data/wujialu/scFM-Bench/output

dataset_name=HLCA_core # scbert, sccello
label_col=cell_type
batch_col=dataset

# dataset_name=Tabula_Sapiens_all # scbert, sccello
# label_col=cell_ontology_class_new
# batch_col=tissue_in_publication

# dataset_name=AIDA_v2_new # scbert, sccello
# label_col=cell_type
# batch_col=donor_id

layer_key=X # for HVG selection

# for model in scCello #scBERT scCello HVG Seurat_cca Harmony scVI Geneformer scGPT UCE xTrimoGene LangCell scCello
# do
#     python 3_cell_clustering.py \
#         --data_folder ${data_folder} \
#         --output_folder ${output_folder} \
#         --model_name ${model} \
#         --dataset_name ${dataset_name} \
#         --label_col ${label_col} \
#         --batch_col ${batch_col} \
#         --layer_key ${layer_key} 
# done

repo_folder=$(dirname "$(dirname "$(readlink -f "$0")")")
echo $repo_folder
pushd $repo_folder/scGraph
    python scGraph_cl_ontology.py $dataset_name 
popd