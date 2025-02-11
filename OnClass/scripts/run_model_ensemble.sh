device_id=7
repo_folder=$(dirname "$(dirname "$(readlink -f "$0")")")
echo $repo_folder
pushd $repo_folder > /dev/null
    python run_one_dataset_cross_validation.py cuda:${device_id} HLCA_core Tabula_Sapiens_all
    python run_one_dataset_cross_validation.py cuda:${device_id} Tabula_Sapiens_all HLCA_core
popd > /dev/null