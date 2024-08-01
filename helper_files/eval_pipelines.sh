#!/bin/bash

# Define the Python file to run
PYTHON_FILE="eval.py"

# Diarized pipelines (recently)
diar_pipelines=(
    "/archive/shared/sim_center/shared/annie/diarized/claude-sonnet-3.5-all/text"
    "/archive/shared/sim_center/shared/annie/diarized/claude-sonnet-3.5-diar/text"
    "/archive/shared/sim_center/shared/annie/diarized/gpt4-all/text"
    "/archive/shared/sim_center/shared/annie/diarized/gpt4-diar/text"
    "/archive/shared/sim_center/shared/annie/diarized/gpt4-sum-gpt4o-diar/text"
    "/archive/shared/sim_center/shared/annie/diarized/llama-8b-all/text"
    "/archive/shared/sim_center/shared/annie/diarized/llama-8b-diar/text"
    "/archive/shared/sim_center/shared/annie/diarized/llama-70b-all/text"
    "/archive/shared/sim_center/shared/annie/diarized/llama-70b-diar/text"
)

next_pipelines=(
    "/archive/shared/sim_center/shared/annie/diarized/gpt4o-all/text"
    "/archive/shared/sim_center/shared/annie/diarized/gpt4o-diar/text"
    "/archive/shared/sim_center/shared/annie/pilot_experiments/model-transcript-folders/gpt4_3chunk_pass2"
    "/archive/shared/sim_center/shared/annie/pilot_experiments/model-transcript-folders/gpt4-3chunk-nosum"
    "/archive/shared/sim_center/shared/annie/pilot_experiments/model-transcript-folders/gpt4-1chunk"
    "/archive/shared/sim_center/shared/annie/pilot_experiments/model-transcript-folders/gpt4-1chunk-nosum"
    "/archive/shared/sim_center/shared/annie/pilot_experiments/model-transcript-folders/GPT4 6-chunk"
    "/archive/shared/sim_center/shared/annie/pilot_experiments/model-transcript-folders/GPT4 9-chunk"
    "/archive/shared/sim_center/shared/annie/pilot_experiments/model-transcript-folders/claude-opus"
    "/archive/shared/sim_center/shared/annie/pilot_experiments/model-transcript-folders/claude-opus-nosum"
    "/archive/shared/sim_center/shared/annie/pilot_experiments/model-transcript-folders/claude-sonnet"
    "/archive/shared/sim_center/shared/annie/pilot_experiments/model-transcript-folders/gpt4o"
    "archive/shared/sim_center/shared/annie/pilot_experiments/model-transcript-folders/gpt4o-nosum"
)

# Run names
run_names=(
    "claude-sonnet-3.5-all"
    "claude-sonnet-3.5-diar"
    "gpt4-all"
    "gpt4-diar"
    "gpt4-sum-gpt4o-diar"
    "llama-8b-all"
    "llama-8b-diar"
    "llama-70b-all"
    "llama-70b-diar"
)

next_names=(
    "gpt4o-all"
    "gpt4o-diar"
    "gpt4_3chunk_pass2"
    "gpt4-3chunk-nosum"
    "gpt4-1chunk"
    "gpt4-1chunk-nosum"
    "GPT4 6-chunk"
    "GPT4 9-chunk"
    "claude-opus"
    "claude-opus-nosum"
    "claude-sonnet"
    "gpt4o"
    "gpt4o-nosum"
)

# Define data path and output directory
data_path="helper_files/all-ids.jsonl"
output_dir="/archive/shared/sim_center/shared/annie/diff_lev_scores"

# Loop through each argument and run the Python file
for i in "${!next_pipelines[@]}"; do
    arg1="${next_pipelines[$i]}"
    arg2="${next_names[$i]}"
    echo "Running: python $PYTHON_FILE --data_path $data_path --output_dir $output_dir --diarized_path_txt \"$arg1\" --run_name \"$arg2\""
    python "$PYTHON_FILE" --data_path "$data_path" --output_dir "$output_dir" --diarized_path_txt "$arg1" --run_name "$arg2"
done

