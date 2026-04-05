export CUDA_VISIBLE_DEVICES=2

# python unsloth_finetune.py \
#     --model_path $MODEL_PATH \
#     --data_path $DATA_PATH \
#     --output_dir $OUTPUT_PATH \
#     --data_split retain50


DATA_PATH="mbpp"
# OUTPUT_PATH="gwen2.5-coder-7b-instruct-finetune-mbpp"
# MODEL_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"

# OUTPUT_PATH="yi-coder-9b-finetune-mbpp"
# MODEL_PATH="01-ai/Yi-Coder-9B"

OUTPUT_PATH="gwen2.5-coder-7b-finetune-mbpp"
MODEL_PATH="Qwen/Qwen2.5-Coder-7B"


# OUTPUT_PATH="deepseek-coder-6.7b-finetune-mbpp"
# MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"

python finetune.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
