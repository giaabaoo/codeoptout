#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
PYTHONPATH='.':$PYTHONPATH \

python3 main.py \
--config configs/datasets/code_split/mbpp_filtered_deepseek.yml \
configs/evaluators/code_evaluator.yml \
configs/models/deepseek/deepseek.yml \
configs/pipelines/evaluate_takedown_at_training.yml \
configs/takedown_methods/ke.yml 
