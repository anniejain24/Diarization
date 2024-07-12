
- [usage](#usage)
  - [set up environment](#set-up-environment)
  - [input data](#input-data)
- [format and where they are from](#format-and-where-they-are-from)
  - [creating promptflow connection](#creating-promptflow-connection)
- [link to promptflow git](#link-to-promptflow-git)
- [describe briefly](#describe-briefly)
  - [promptflow pipelines](#promptflow-pipelines)
- [describe each one](#describe-each-one)
  - [run a test promptflow](#run-a-test-promptflow)
  - [run a batch promptflow](#run-a-batch-promptflow)
- [describe jsonl](#describe-jsonl)
  - [run python model](#run-python-model)

# usage
hello `run.py`
## set up environment

```shell

conda env create -f diarize_env.yaml -y
conda activate diarize

```
## input data
# format and where they are from

## creating promptflow connection
# link to promptflow git
# describe briefly

## promptflow pipelines
# describe each one

## run a test promptflow

```shell
export BASE_FLOW_PATH="/work/bioinformatics/s229618/diarization/"
# edit path with flow name [e.g. prompt_diarize_3chunk, prompt_diarize_1chunk]
python -m promptflow._cli._pf.entry flow test --flow $BASE_FLOW_PATH/prompt_diarize_3chunk

```

## run a batch promptflow
# describe jsonl

```shell
export BASE_FLOW_PATH="/work/bioinformatics/s229618/diarization/"
# edit path with flow name [e.g. prompt_diarize_3chunk, prompt_diarize_1chunk]
# change below to batch command
python -m promptflow._cli._pf.entry flow test --flow $BASE_FLOW_PATH/prompt_diarize_3chunk

```

## run python model

```shell
export BASE_FLOW_PATH="/work/bioinformatics/s229618/diarization/"
# edit path with flow name [e.g. prompt_diarize_3chunk, prompt_diarize_1chunk]
# change below to batch command
python run.py --config_path config.yaml

```