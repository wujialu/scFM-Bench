device_id=0

dataset=HLCA_core
batch_col=dataset

# dataset=Tabula_Sapiens_all
# batch_col=tissue_in_publication

# dataset=AIDA_v2_new
# batch_col=donor_id

repo_folder=$(dirname "$(dirname "$(readlink -f "$0")")")
echo $repo_folder
pushd $repo_folder > /dev/null
    for model in scCello
    do
        python run_one_dataset_cv_optuna.py cuda:${device_id} ${model} ${dataset} ${batch_col}
    done     
popd > /dev/null