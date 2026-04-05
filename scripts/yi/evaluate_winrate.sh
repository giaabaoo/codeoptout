#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
PYTHONPATH='.':$PYTHONPATH \

python3 main.py \
--config configs/datasets/code_split/mbpp_filtered_yi.yml \
configs/evaluators/winrate_evaluator.yml \
configs/models/yi/yi.yml \
configs/pipelines/evaluate_winrate.yml \
