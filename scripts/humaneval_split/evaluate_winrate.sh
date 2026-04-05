#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/code_split/humaneval.yml \
configs/evaluators/winrate_evaluator.yml \
configs/models/deepseek/deepseek.yml \
configs/pipelines/evaluate_winrate.yml \
