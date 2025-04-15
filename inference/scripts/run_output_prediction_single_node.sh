#!/bin/bash

dirs=(
    "deepseek-r1-distill-qwen-1.5b"
    "semcoder-s-1030"
    "semcoder-1030"
    "deepseek-coder-7b-instruct-v1.5"
    "starcoder2-7b"
    "qwen2.5-coder-7b"
    "qwen2.5-coder-7b-instruct"
    "magicoder-cl-7b"
    "magicoder-s-cl-7b"
    "codellama-7b-python"
    "codellama-7b"
    "magicoder-ds-6.7b"
    "magicoder-s-ds-6.7b"
    "deepseek-coder-6.7b-base"
    "deepseek-coder-6.7b-instruct"
    "qwen2.5-coder-14b"
    "qwen2.5-coder-14b-instruct"
    "deepseek-coder-v2-lite-base"
    "deepseek-coder-v2-lite-instruct"
    "starcoder2-15b-instruct-v0.1"
    "codellama-13b-python"
    "codellama-13b"
    "starcoder2-15b"
    "deepseek-r1-distill-qwen-32b"
)

models=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "semcoder/semcoder_s_1030"
    "semcoder/semcoder_1030"
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    "bigcode/starcoder2-7b"
    "Qwen/Qwen2.5-Coder-7B"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    "ise-uiuc/Magicoder-CL-7B"
    "ise-uiuc/Magicoder-S-CL-7B"
    "codellama/CodeLlama-7b-Python-hf"
    "codellama/CodeLlama-7b-hf"
    "ise-uiuc/Magicoder-DS-6.7B"
    "ise-uiuc/Magicoder-S-DS-6.7B"
    "deepseek-ai/deepseek-coder-6.7b-base"
    "deepseek-ai/deepseek-coder-6.7b-instruct"
    "Qwen/Qwen2.5-Coder-14B"
    "Qwen/Qwen2.5-Coder-14B-Instruct"
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    "bigcode/starcoder2-15b-instruct-v0.1"
    "codellama/CodeLlama-13b-Python-hf"
    "codellama/CodeLlama-13b-hf"
    "bigcode/starcoder2-15b"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

temperatures=(0.2 0.8)

for ((i=0; i<${#models[@]}; i++)); do
    model=${models[$i]}
    base_dir=${dirs[$i]}
    echo "$model"
    for temperature in "${temperatures[@]}"; do
        dir="${base_dir}_temp${temperature}_output"
        echo "Starting for $model at $temperature"
        mkdir -p model_generations_raw/"$dir"
        python main.py \
            --model "$model" \
            --use_auth_token \
            --trust_remote_code \
            --tasks output_prediction \
            --batch_size 10 \
            --n_samples 10 \
            --max_length_generation 1024 \
            --precision fp16 \
            --limit 800 \
            --temperature "$temperature" \
            --save_generations \
            --save_generations_path model_generations_raw/"$dir"/generations.json \
            --start 0 \
            --end 800 \
            --tensor_parallel_size 1
    done
done