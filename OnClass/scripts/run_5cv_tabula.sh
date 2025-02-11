device_id=0
repo_folder=$(dirname "$(dirname "$(readlink -f "$0")")")
echo $repo_folder
pushd $repo_folder > /dev/null
    #! step1: train cross-validation models
    for model in scVI Geneformer scGPT UCE LangCell xTrimoGene
    do
        python run_one_dataset_cross_validation.py cuda:${device_id} ${model} Tabula_Sapiens_all
    done

    #! step2: inference on non-leaf nodes 
    for model in scVI Geneformer scGPT UCE LangCell xTrimoGene
    do
        python run_inference.py cuda:${device_id} ${model} Tabula_Sapiens_all Tabula_Sapiens_all
        python run_inference.py cuda:${device_id} ${model} Tabula_Sapiens_all HLCA_core 
    done

    #! step3: calculate lcad and save in pred_label.csv
    python evaluate_lcad.py Tabula_Sapiens_all Tabula_Sapiens_all
    python evaluate_lcad.py Tabula_Sapiens_all HLCA_core
popd > /dev/null