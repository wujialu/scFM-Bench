device_id=0
repo_folder=$(dirname "$(dirname "$(readlink -f "$0")")")
echo $repo_folder
pushd $repo_folder > /dev/null
    # python run_LR_OvR.py AIDA_v2_new
    # python run_LR_OvR.py HLCA_core
    # python run_LR_OvR.py Tabula_Sapiens_all
    
    python evaluate_lcad.py AIDA_v2_new AIDA_v2_new
    python evaluate_lcad.py HLCA_core HLCA_core
    python evaluate_lcad.py Tabula_Sapiens_all Tabula_Sapiens_all
    python evaluate_lcad.py Tabula_Sapiens_all HLCA_core
    python evaluate_lcad.py HLCA_core Tabula_Sapiens_all
popd > /dev/null