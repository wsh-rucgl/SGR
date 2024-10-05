model_name='XXX'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch pretrain4SGR.py \
   --do_train \
   --output_dir ./output/${model_name}  \
   --overwrite_output_dir True \
   --num_train_epochs 2 \
   --per_device_train_batch_size 2 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --logging_steps 10 \
   --save_safetensors False \
   --learning_rate 5e-5 \
   --save_steps 10000 \
   --seed 0 \
   --bf16 \
   --report_to none 

# nohup bash pretrain4SGR.sh &>XXX.log &
