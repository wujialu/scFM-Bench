device_id=1
model_loc=./data/weights/UCE/33l_8ep_1024t_1280.torch
path_to_anndata=./data/datasets/pancreas_scib.h5ad
species=human
output_dir=./output/UCE/

mkdir -p ${output_dir}

workdir=../../UCE
cd $workdir

CUDA_VISIBLE_DEVICES=${device_id} python eval_single_anndata.py \
    --adata_path ${path_to_anndata} \
    --dir ${output_dir} \
    --species ${species} \
    --model_loc ${model_loc} --nlayers 33 \
