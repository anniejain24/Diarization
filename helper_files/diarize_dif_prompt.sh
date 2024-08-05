#!/bin/bash

# Define data path and output directory
data_path="helper_files/all-ids.jsonl"
output_dir="/archive/shared/sim_center/shared/annie/diarized/"

# change here and below. Make sure to set env variables and other config
model_name=(
    claude-sonnet-3.5
    azure
    llama-70b
)

azure_mod=gpt4-feb

diarize_path=(
    helper_files/diarize_prompt_zero_shot.txt
    helper_files/diarize_prompt_few_shot.txt
    helper_files/diarize_prompt_whole_shot.txt
)

echo "python run.py --data_path $data_path --output_dir ${output_dir}${azure_mod}-0sh --diar_model ${model_name[1]} --diarize_path ${diarize_path[0]}"
python run.py --data_path "$data_path" --output_dir "${output_dir}${azure_mod}-0sh" --diar_model "${model_name[1]}" --diarize_path "${diarize_path[0]}"
echo "python run.py --data_path $data_path --output_dir ${output_dir}${azure_mod}-fsh --diar_model ${model_name[1]} --diarize_path ${diarize_path[1]}"
python run.py --data_path "$data_path" --output_dir "${output_dir}${azure_mod}-fsh" --diar_model "${model_name[1]}" --diarize_path "${diarize_path[1]}"
echo "python run.py --data_path $data_path --output_dir ${output_dir}${azure_mod}-wsh --diar_model ${model_name[1]} --diarize_path ${diarize_path[2]}"
python run.py --data_path "$data_path" --output_dir "${output_dir}${azure_mod}-wsh" --diar_model "${model_name[1]}" --diarize_path "${diarize_path[2]}"