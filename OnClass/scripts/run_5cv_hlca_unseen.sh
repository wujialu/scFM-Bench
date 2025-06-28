device_id=2
repo_folder=$(dirname "$(dirname "$(readlink -f "$0")")")
echo $repo_folder
pushd $repo_folder > /dev/null
    for model in Harmony Seurat_cca #scVI Geneformer scGPT UCE LangCell xTrimoGene
    do
        python run_one_dataset_cv_unseen.py cuda:${device_id} ${model} HLCA_core
    done
popd > /dev/null