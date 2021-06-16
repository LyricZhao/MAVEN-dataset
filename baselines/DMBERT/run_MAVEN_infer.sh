export OUTPUT_PATH=./MAVEN_deberta
export CK_NUM=1
export MODEL_NAME=deberta
python3.6 run_ee.py \
    --data_dir /home/zhaocg/MAVEN/ \
    --model_type ${MODEL_NAME} \
    --model_name_or_path ${OUTPUT_PATH}/checkpoint-${CK_NUM} \
    --task_name maven_infer \
    --output_dir ${OUTPUT_PATH} \
    --max_seq_length 128 \
    --do_lower_case \
    --per_gpu_train_batch_size 84 \
    --per_gpu_eval_batch_size 84 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_infer
python3.6 get_submission.py \
    --test_data /home/zhaocg/MAVEN/test.jsonl \
    --preds ${OUTPUT_PATH}/checkpoint-${CK_NUM}/checkpoint-${CK_NUM}_preds.npy \
    --output ${OUTPUT_PATH}/results.jsonl
