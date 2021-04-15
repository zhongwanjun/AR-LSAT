export LSAT_DIR=../data/model_data
export TEST_FILE=../model_data/ar_val_analyze_condition.json
export TRAIN_FILE=ar_train_analyze_condition.json
export VAL_FILE=ar_val_analyze_condition.json
export TASK_NAME=lsat
export LEARNING_RATE=1e-5
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
    --do_lower_case \
    --data_dir $LSAT_DIR \
    --train_file $TRAIN_FILE \
    --dev_file $VAL_FILE \
    --test_file $TEST_FILE \
    --gradient_accumulation_steps 1 \
    --save_steps 200 \
    --adam_betas "(0.9, 0.98)" \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs 50.0 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --overwrite_output_dir \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size 1  \
    --per_gpu_train_batch_size 4    \
    --output_dir ../../Checkpoints/$TASK_NAME/single_GRU_test  \
    --logging_steps 200 \
    --local_rank -1 \
    --overwrite_cache
