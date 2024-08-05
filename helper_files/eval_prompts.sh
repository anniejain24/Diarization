#!/bin/bash

# Define the Python file to run
PYTHON_FILE="eval.py"

diar_path=/archive/shared/sim_center/shared/annie/diarized/
# Diarized pipelines (or insert desired ones and run names below)
diar_pipelines=(
    claude-sonnet-3.5-0sh
    claude-sonnet-3.5-fsh
    claude-sonnet-3.5-wsh
    gpt4o-0sh
    gpt4o-fsh
    gpt4o-wsh
    llama-70b-0sh
    llama-70b-wsh
    llama-70b-fsh
)

more_pipelines=(
    gpt4-feb-0sh
    gpt4-feb-fsh
    gpt4-feb-wsh
)

# Define data path and output directory
data_path="helper_files/all-ids.jsonl"
output_dir="/archive/shared/sim_center/shared/annie/diff_lev_scores"

# Loop through each argument and run the Python file
# Add a /text if needed
for i in "${!more_pipelines[@]}"; do
    arg1="${more_pipelines[$i]}"
    echo "Running: python $PYTHON_FILE --data_path $data_path --output_dir $output_dir --diarized_path_txt \"$diar_path$arg1\" --run_name \"$arg1\""
    python "$PYTHON_FILE" --data_path "$data_path" --output_dir "$output_dir" --diarized_path_txt "$diar_path$arg1" --run_name "$arg1"
done