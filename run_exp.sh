#!/bin/bash

python inference/main.py \
    --model "Qwen/Qwen2.5-Coder-1.5B" \
    --use_auth_token \
    --trust_remote_code \
    --tasks output_prediction \
    --batch_size 1 \
    --n_samples 1 \
    --max_length_generation 1024 \
    --precision fp16 \
    --limit 800 \
    --temperature 0 \
    --save_generations \
    --save_generations_path model_generations_raw/qwen1/generations.json \
    --start 0 \
    --end 800 \
    --tensor_parallel_size 1