#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/code_split/humaneval.yml \
configs/evaluators/code_evaluator.yml \
configs/models/deepseek/deepseek.yml \
configs/pipelines/evaluate_vanilla.yml
