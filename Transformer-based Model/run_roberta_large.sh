export LSAT_DIR=../data
export TASK_NAME=lsat
export MODEL_NAME=roberta-base
#python main_large.py \
#python -m torch.distributed.launch --nproc_per_node=2 main_large.py \
python main_large.py \
    --model_type roberta \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --do_test \
    --data_dir $LSAT_DIR \
    --gradient_accumulation_steps 1 \
    --save_steps 200 \
    --adam_betas "(0.9, 0.98)" \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --learning_rate 1e-5 \
    --num_train_epochs 15.0 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --overwrite_output_dir \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 4   \
    --per_gpu_train_batch_size 4    \
    --output_dir ../Checkpoints/$TASK_NAME/roberta-large_test  \
    --logging_steps 200 \
    --overwrite_cache \
    --local_rank -1 \
