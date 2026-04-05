#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
PYTHONPATH='.':$PYTHONPATH \

python3 main.py \
--config configs/datasets/code_split/mbpp_filtered_qwen.yml \
configs/evaluators/code_evaluator.yml \
configs/models/gwen/gwen.yml \
configs/pipelines/evaluate_takedown_at_decoding.yml \
configs/takedown_methods/r_cad.yml # 
