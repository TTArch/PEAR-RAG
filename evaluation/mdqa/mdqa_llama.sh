#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <start_index> <end_index> <cuda_index>"
    exit 1
fi

START_INDEX=$1
END_INDEX=$2
CUDA_INDEX=$3
MODEL_NAME="longma2-5"

IDS=(1 3 5 7 10)

MODEL_PATH="./meta-llama/Llama-2-7b-chat-hf"
TEST_DATA_PATH="./mdqa/mdqa/qa_data/10_total_documents/mdqa_10documents.jsonl"

export CUDA_VISIBLE_DEVICES=$CUDA_INDEX

for ID in "${IDS[@]}"; do
    OUTPUT_FILE="test_mdqa/ablation_llama2_trained_${MODEL_NAME}/output_folder_${ID}"

    if [ ! -d "$OUTPUT_FILE" ]; then
        mkdir -p "$OUTPUT_FILE"
    fi

    python pear_mdqa_test.py \
        --model_name_or_path "$MODEL_PATH" \
        --test_data_path "$TEST_DATA_PATH" \
        --bf16 True \
        --correct_doc_id "$ID" \
        --output_dir "$OUTPUT_FILE" \
        --overwrite_output_dir True \
        --save_safetensors False \
        --start "$START_INDEX" \
        --end "$END_INDEX" \
        --model_name "$MODEL_NAME" \
        --source_model_max_length 4096 \
        --model_max_length 4096
done