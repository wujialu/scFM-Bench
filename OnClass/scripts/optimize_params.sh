device_id=1
# dataset=HLCA_core
dataset=Tabula_Sapiens_all

repo_folder=$(dirname "$(dirname "$(readlink -f "$0")")")
echo $repo_folder
pushd $repo_folder > /dev/null
    for model in Harmony Seurat_cca #scVI Geneformer scGPT UCE LangCell xTrimoGene
    do
        python run_one_dataset_cv_optuna.py cuda:${device_id} ${model} ${dataset}
    done     
popd > /dev/null