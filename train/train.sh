model_name='ablation_40_negfilter_50000_train_epoch1'

CUDA_VISIBLE_DEVICES=0 python train.py \
   --do_train \
   --output_dir ./pear_llama_ckpt/${model_name} \
   --overwrite_output_dir True \
   --num_train_epochs 1 \
   --lr_scheduler_type linear \
   --per_device_train_batch_size 1 \
   --learning_rate 0.005 \
   --gradient_accumulation_steps 20 \
   --warmup_steps 2 \
   --save_steps 10 \
   --save_total_limit 250 \
   --save_only_model True \
   --logging_steps 25 \
   --save_safetensors False \
   --bf16 \
   --seed 0 \
   --report_to none