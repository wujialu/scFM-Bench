# dataset_name=HLCA_core
# label_col=cell_type
# batch_col=dataset

dataset_name=Tabula_Sapiens_all
label_col=cell_ontology_class_new
batch_col=tissue_in_publication

layer_key=X # for HVG selection

# for model in Seurat_cca Harmony #HVG scVI Geneformer scGPT UCE xTrimoGene LangCell
# do
#     python 3_cell_clustering.py \
#         --model_name ${model} \
#         --dataset_name ${dataset_name} \
#         --label_col ${label_col} --batch_col ${batch_col} --layer_key ${layer_key} 
# done

repo_folder=$(dirname "$(dirname "$(readlink -f "$0")")")
echo $repo_folder
pushd $repo_folder/scGraph
    python scGraph_cl_ontology.py $dataset_name
popd