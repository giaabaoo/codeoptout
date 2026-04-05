#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/code_split/mbpp_gsm8k.yml \
configs/evaluators/code_evaluator.yml \
configs/models/deepseek/deepseek.yml \
configs/pipelines/evaluate_takedown_at_inference.yml \
configs/takedown_methods/sys_prompt.yml # general
