data_path=./train_data/normal_data
proxy_data_path=./train_data/proxy_data

overwrite=1

ID_emb_model_path=./model/arcface/ms1mv3_arcface_r100_fp16_backbone.pth

run_id=base
device=cuda:0

expr_path=./expr/train_smswap_faceshiter_adapter
config_name=train_config_bsz8_faceshifter

cd train

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29500 train_adapter.py \
    --data_path $data_path \
    --data_ssl_path $proxy_data_path \
    --expr_path $expr_path \
    --ID_emb_model_path $ID_emb_model_path \
    --config_name $config_name --device $device --overwrite $overwrite --run_id $run_id \
    --save_models 0 \
    --resume 1 \
    --epoch 33 \
    --adapter_type add