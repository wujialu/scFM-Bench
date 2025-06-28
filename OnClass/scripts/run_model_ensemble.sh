device_id=2
repo_folder=$(dirname "$(dirname "$(readlink -f "$0")")")
echo $repo_folder
pushd $repo_folder > /dev/null
    python run_inference_ensemble.py cuda:${device_id} HLCA_core Tabula_Sapiens_all
    python run_inference_ensemble.py cuda:${device_id} Tabula_Sapiens_all HLCA_core
popd > /dev/nulls