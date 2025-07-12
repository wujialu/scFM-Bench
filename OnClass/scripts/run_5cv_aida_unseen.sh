device_id=0
repo_folder=$(dirname "$(dirname "$(readlink -f "$0")")")
echo $repo_folder
pushd $repo_folder > /dev/null
    for model in xTrimoGene #Harmony Seurat_cca scVI Geneformer scGPT UCE LangCell xTrimoGene scCello
    do
        python run_one_dataset_cv_unseen.py cuda:${device_id} ${model} AIDA_v2_new donor_id
    done
popd > /dev/null