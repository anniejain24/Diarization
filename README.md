
- [**Overview**](#overview)
  - [diarization](#diarization)
  - [types of models and possible pipelines](#types-of-models-and-possible-pipelines)
  - [evaluation and metrics](#evaluation-and-metrics)
- [Diarization usage](#diarization-usage)
  - [set up environment](#set-up-environment)
  - [input data](#input-data)
  - [set azure environment variables](#set-azure-environment-variables)
  - [run a model](#run-a-model)

# <span style="color:purple">**Overview**</span>

## diarization

## structuring of input data

Input data should be provided in a json format, with a list of elements, each element containing a "text" field. E.g: files are outputted from whisper.ai in this way, with corresponding text and timestamps. Any other fields can be provided as well, but the text field for each utterance is necessary. 

![alt text](github/diarization/images/input_format.png "Input format")

If the transcript is simply a string, use format.py to perform sentence tokenization and output the life-formatted json version



## types of models and possible pipelines

Models:

llama:

- llama 8 billion
- llama 70 billion

claude:

- claude sonnet 3 or 3.5
- claude opus 3
- claude haiku 3

azure:

- any deployment which is hosted on azure (e.g. gpt4o, gpt4, gpt3.5, gpt3.5 turbo, mixtral, etc.)

## evaluation and metrics

*creation of gold-standard set:*

UNDER CONSTRUCTION


# Diarization usage
All pipelines can be run using `run.py` 
Customize by editing `config.yaml` with the desired models, pipeline components, and parameters

## set up environment

```shell
conda env create -f diarize_env.yaml -y
conda activate diarize
```
## input data
format and where they are from


## set azure environment variables
In order to run an azure deployment, you will have to set the following environment variables by using the following command line prompts

```shell
export OPENAI_API_VERSION="2024-05-01-preview" (or other model version)
export AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com/"
export AZURE_OPENAI_API_KEY= "<your-api-key>"
```

## run a model

--config is required argument, and default configs are stored in `config.yaml`:

```shell
python run.py --config config.yaml 
```

*description of configs and allowed values:*

**temperature:**

**top_p:**

**max_new_tokens:**

**do_sample:**

**mod_name:**


**OPTIONAL** With additional optional args (change paths to desired paths):

```shell
python run.py --config config.yaml --data_path 'helper_files/test-id.jsonl' --summary_path 'helper_files/summary_prompt.txt' --diarize_path 'helper_files/diarize_prompt.txt' --output_dir '<insert alternative path>'
```
**data_path:** path to jsonl file with IDs to diarize (see `helper_files/test-id.jsonl`, or `helper_files/ids.jsonl` for longer batch)

**summary_path:** path to summary prompt, default in `helper_files/summary_prompt.txt`

**diarize_path:** path to diarization prompt, default in `helper_files/diarize_prompt.txt`

**output_dir:** path to directory to save diarization output, default saves tp `temp/`


[def]: github/diarization/images/input_format.png