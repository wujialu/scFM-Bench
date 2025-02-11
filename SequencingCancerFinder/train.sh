train_dir=../output/TISCH
val_dir=../output/TISCH
val_domain=Bone
input_features=UCE
max_epoch=200
patience=20
label_str=label
gpu_id=4
dropout=0.1
weight_decay=1e-6
lr=1e-3
batch_size=128
NEED_ROWS=20000
gene_num=5000

for val_domain in Blood Bone Brain Eye
do
output=results/${val_domain}/ELU_log_100epoch_20250116
# for input_features in scVI_integrated_Patient scVI_integrated_dataset scVI_integrated_tumor scVI_integrated_tissue  
for input_features in X Geneformer LangCell scGPT UCE xTrimoGene
do 
    python train.py --gpu_id ${gpu_id} \
        --val_domain ${val_domain} \
        --train_dir ${train_dir} \
        --val_dir ${val_dir} \
        --input_features ${input_features} \
        --lr ${lr} \
        --max_epoch ${max_epoch} --patience ${patience} \
        --dropout ${dropout} --weight_decay ${weight_decay} \
        --batch_size ${batch_size} \
        --label_str ${label_str} \
        --output ${output} \
        --NEED_ROWS ${NEED_ROWS} \
        --gene_num ${gene_num} \
        --batch_size ${batch_size} \
        --num_classes 1 
done
done