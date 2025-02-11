device_id=7
repo_folder=$(dirname "$(dirname "$(readlink -f "$0")")")
echo $repo_folder
pushd $repo_folder > /dev/null
    #! step1: train cross-validation models
    for model in scVI Geneformer scGPT UCE LangCell xTrimoGene
    do
        python run_one_dataset_cross_validation.py cuda:${device_id} ${model} HLCA_core
    done

    #! step2: inference on non-leaf nodes 
    for model in scVI Geneformer scGPT UCE LangCell xTrimoGene
    do
        python run_inference.py cuda:${device_id} ${model} HLCA_core HLCA_core
        python run_inference.py cuda:${device_id} ${model} HLCA_core Tabula_Sapiens_all
    done

    #! step3: calculate lcad and save in pred_label.csv
    python evaluate_lcad.py HLCA_core HLCA_core
    python evaluate_lcad.py HLCA_core Tabula_Sapiens_all
popd > /dev/null