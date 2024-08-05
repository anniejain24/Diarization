#!/bin/bash

# Define the Python file to run
PYTHON_FILE="eval.py"

diar_path=/archive/shared/sim_center/shared/annie/diarized/
# Diarized pipelines (or insert desired ones and run names below)
diar_pipelines=(
    claude-sonnet-3.5-9ch
)

segments=(
    3
    6
    9
    20
)

# Define data path and output directory
data_path="helper_files/all-ids.jsonl"
output_dir="/archive/shared/sim_center/shared/annie/diff_lev_scores"

# Loop through each argument and run the Python file
for i in "${!segments[@]}"; do
    arg1="${diar_pipelines[0]}"
    arg2="${segments[$i]}"
    echo "Running: python $PYTHON_FILE --data_path $data_path --output_dir $output_dir --diarized_path_txt \"$diar_path$arg1/text\" --run_name \"$arg1-$arg2-seg\" --segments $arg2"
    python "$PYTHON_FILE" --data_path "$data_path" --output_dir "$output_dir" --diarized_path_txt "$diar_path$arg1/text" --run_name "$arg1-$arg2-seg" --segments $arg2
done