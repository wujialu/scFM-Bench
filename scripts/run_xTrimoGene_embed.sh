device_id=0
task_name=pancreas_scib
input_type=singlecell
output_type=cell
pool_type=all
tgthighres=f1
data_path=./data/datasets/xTrimoGene/pancreas_scib_19264.h5ad
save_path=./output/xTrimoGene/
pre_normalized=F # false
version=ce

mkdir -p ${save_path}

workdir=../../xTrimoGene/model
cd $workdir

CUDA_VISIBLE_DEVICES=${device_id} python get_embedding.py \
    --task_name ${task_name} \
    --input_type ${input_type} \
    --output_type ${output_type} \
    --pool_type ${pool_type} \
    --tgthighres ${tgthighres} \
    --data_path ${data_path} \
    --save_path ${save_path} \
    --pre_normalized ${pre_normalized} \
    --version ${version}