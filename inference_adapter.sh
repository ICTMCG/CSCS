ID_emb_model_path=./model/arcface/ms1mv3_arcface_r100_fp16_backbone.pth

adapter_type=add
model_name=model_34_loss_-0.1688.pth.tar
weight_path=./${model_name}


src_path=/path/to/aligned/src/image
tgt_path=/path/to/aligned/tgt/image
output_dir=./output

echo $weight_path

CUDA_VISIBLE_DEVICES=0 python inference_adapter.py \
    --adapter_type $adapter_type \
    --src_path $src_path \
    --tgt_path $tgt_path \
    --output_dir $output_dir \
    --weight_path $weight_path \
    --ID_emb_model_path $ID_emb_model_path