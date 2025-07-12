device_id=0
repo_folder=$(dirname "$(dirname "$(readlink -f "$0")")")
echo $repo_folder
pushd $repo_folder > /dev/null
    # #! step1: train cross-validation models
    # for model in xTrimoGene #Harmony Seurat_cca scVI Geneformer scGPT UCE LangCell xTrimoGene scCello
    # do
    #     python run_one_dataset_cross_validation.py cuda:${device_id} ${model} AIDA_v2_new donor_id
    # done

    # #! step2: run inference
    # for model in xTrimoGene #Harmony Seurat_cca scVI Geneformer scGPT UCE LangCell xTrimoGene scCello
    # do
    #     # same dataset: inference on nonleaf nodes (awareness of cell type hierarchies)
    #     python run_inference.py cuda:${device_id} ${model} AIDA_v2_new AIDA_v2_new
    #     # different dataset: inter-dataset validation (generalization ability)
    #     # python run_inference.py cuda:${device_id} ${model} AIDA_v2_new Tabula_Sapiens_all
    # done

    #! step3: calculate lcad and save in pred_label.csv
    python evaluate_lcad.py AIDA_v2_new AIDA_v2_new
    # python evaluate_lcad.py AIDA_v2_new Tabula_Sapiens_all
popd > /dev/null