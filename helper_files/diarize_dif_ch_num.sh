#!/bin/bash

# Define data path and output directory
data_path="helper_files/all-ids.jsonl"
output_dir="/archive/shared/sim_center/shared/annie/diarized/"

# change here and below. Make sure to set env variables and other config
model_name=(
    llama-70b
    claude-sonnet-3.5
    azure
)

echo "python run.py --data_path $data_path --output_dir ${output_dir}${model_name[0]}-3ch --diar_model ${model_name[0]} --chunk_num 3"
python run.py --data_path "$data_path" --output_dir "${output_dir}${model_name[0]}-3ch" --diar_model "${model_name[0]}" --chunk_num 3
echo "python run.py --data_path $data_path --output_dir ${output_dir}${model_name[0]}-6ch --diar_model ${model_name[0]} --chunk_num 6"
python run.py --data_path "$data_path" --output_dir "${output_dir}${model_name[0]}-6ch" --diar_model "${model_name[0]}" --chunk_num 6
echo "python run.py --data_path $data_path --output_dir ${output_dir}${model_name[0]}-9ch --diar_model ${model_name[0]} --chunk_num 9"
python run.py --data_path "$data_path" --output_dir "${output_dir}${model_name[0]}-9ch" --diar_model "${model_name[0]}" --chunk_num 9



