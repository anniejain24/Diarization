#!/bin/bash

# Define the Python file to run
PYTHON_FILE="eval.py"

diar_path=/archive/shared/sim_center/shared/annie/diarized/
# Diarized pipelines (or insert desired ones and run names below)
diar_pipelines=(
    gpt4o-diar
    claude-sonnet-3.5-3ch
    claude-sonnet-3.5-6ch
    claude-sonnet-3.5-9ch
    gpt4o-3ch
    gpt4o-6ch
    gpt4o-9ch
    llama-70b-3ch
    llama-70b-6ch
    llama-70b-9ch
)

init_path=/archive/shared/sim_center/shared/annie/pilot_experiments/model-transcript-folders
# init pipelines (pilot experiments)
init_pipelines=(
    gpt4o-nosum
)
# Define data path and output directory
data_path="helper_files/all-ids.jsonl"
output_dir="/archive/shared/sim_center/shared/annie/diff_lev_scores"

# Loop through each argument and run the Python file
for i in "${!diar_pipelines[@]}"; do
    arg1="${diar_pipelines[$i]}"
    echo "Running: python $PYTHON_FILE --data_path $data_path --output_dir $output_dir --diarized_path_txt \"$diar_path$arg1/text\" --run_name \"$arg1\""
    python "$PYTHON_FILE" --data_path "$data_path" --output_dir "$output_dir" --diarized_path_txt "$diar_path$arg1/text" --run_name "$arg1"
done

