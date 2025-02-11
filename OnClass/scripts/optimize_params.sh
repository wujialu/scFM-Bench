device_id=0
dataset=HLCA_core
repo_folder=$(dirname "$(dirname "$(readlink -f "$0")")")
echo $repo_folder
pushd $repo_folder > /dev/null
    for model in scVI Geneformer scGPT UCE LangCell xTrimoGene
    do
        python run_one_dataset_cv_optuna.py cuda:${device_id} ${model} ${dataset}
    done     
popd > /dev/null