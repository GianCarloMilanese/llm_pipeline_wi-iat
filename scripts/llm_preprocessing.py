#!/usr/bin/env python3

import argparse
import json
import os
import pandas as pd
from datetime import datetime

from preprocessing_utils import strip_quotes

parser = argparse.ArgumentParser()
parser.add_argument("data", help="path of dataset on which to perform the preprocessing",
                    default="../data/train/semevalironytrainAll_3k.csv")
parser.add_argument("-d", "--directory", help="Output directory",
                    default="./datasets_preprocessed/")
parser.add_argument("-g", "--gpu", help="Which gpu to use", default="cuda:0")
parser.add_argument("-c", "--config", help="path of JSON config file",
                    default="./config/semevalIrony.json")
parser.add_argument("-m", "--model", help="Model name",
                    default="google/gemma-2-9b-it")
parser.add_argument("-l", "--lemmatization",
                    help="whether to perform lemmatization or not",
                    action="store_true")
parser.add_argument("-w", "--stopwords",
                    help="whether to remove stopwords or not", action="store_true")
parser.add_argument("-s", "--stemming",
                    help="whether to perform stemming or not", action="store_true")
# parser.add_argument("--overwrite", help="Wether to overwrite the output file", action="store_true")
parser.add_argument('--overwrite', nargs='+', default=[],
                    help="names of the columns to overwrite, separated by a space")
parser.add_argument("--sample", help="do a trial run with fewer examples", action="store_true")
parser.add_argument("--test", help="test script without LLM", action="store_true")
args = parser.parse_args()

if not args.test:
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline, logging
    )
    logging.set_verbosity_error()
    torch.random.manual_seed(0)
    # torch.set_default_device('cuda')
    torch.cuda.set_device(args.gpu)


columns_to_overwrite = set(args.overwrite)
model_name = args.model

# Load model
if not args.test:
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda",
                                                 torch_dtype="auto", config=model_config,
                                                 trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generation_args = {"max_new_tokens": 800, "temperature": 0.7, "do_sample": True}

with open(args.config, "r") as f:
    config = json.load(f)

# Load data to preprocess
data = pd.read_csv(args.data)

def format_text(template, text):
    return template.format(strip_quotes(text))

def process_texts(texts, preprocessing_type, config=config):
    prompt = config["prompts"]
    prompt_query = prompt[preprocessing_type]['query']

    output=[]
    for text in texts:
        try:
            message = []
            formatted_text = format_text(prompt_query, text)
            message.append({"role": "user", "content": formatted_text})
            if args.test:
                placeholder_text = "\n".join([m["role"]+": "+m["content"] for m in message])
                generated_text = f"PLACEHOLDER {placeholder_text}"
            else:
                generated_text = pipe(message, **generation_args)[0]["generated_text"]
            output.append(generated_text)
        except Exception as e:
            print(e)
    return output


model_basename = model_name.split("/")[-1]
data_basename = os.path.basename(args.data).split('.')[0]
out_data_name = os.path.join(args.directory, f"{data_basename}_{model_basename}.csv")

if "language" in config["prompts"].keys():
    language = config["prompts"]["language"]
    out_data_name = out_data_name.replace(".csv", f"_{language}-prompt_.csv")

if args.test:
    out_data_name = out_data_name.replace(".csv", "_TEST.csv")

if args.sample:
    sample_size = 50
    data = data.sample(frac=1, random_state=0)
    original_texts = data.Text.values[:sample_size]
    out_data_name = out_data_name.replace(".csv", "_SAMPLE.csv")
else:
    original_texts = list(data.Text.values)

if os.path.exists(out_data_name):
    out_data_df = pd.read_csv(out_data_name)
else:
    out_data_df = data
    out_data_df.to_csv(out_data_name, index=None)

columns_to_lemmatize = ["Text"]
columns_to_remove_stopwords = ["Text"]
columns_to_stem = ["Text"]

print(f"Running preprocessing on {data_basename}, with {model_name}")

if args.lemmatization:
    out_data_df = pd.read_csv(out_data_name)
    for column in columns_to_lemmatize:
        now = datetime.now()
        out_column = f"{column}_lemmatized"
        columns_to_remove_stopwords.append(out_column)
        if out_column in out_data_df.columns and out_column not in columns_to_overwrite:
            print(f"\tNot lemmatizing {column}: {out_column} already exists", end=" ")
            print(f"If you want to run the preprocessing again, pass {out_column} to --overwrite")
            continue
        elif out_column in out_data_df.columns and out_column in columns_to_overwrite:
            print(f"\t{now} Lemmatizing {column}: overwriting {out_column}")
        else:
            print(f"\t{now} Lemmatizing {column}")
        preprocessed_texts = process_texts(list(out_data_df[column].values),
                                           preprocessing_type="lemmatization")
        out_data_df[out_column] = preprocessed_texts
        out_data_df.to_csv(out_data_name, index=None)

if args.stopwords:
    out_data_df = pd.read_csv(out_data_name)
    for column in columns_to_remove_stopwords:
        out_columns = []
        preprocessing_types = []
        if args.stopwords:
            out_columns.append(f"{column}_nostopwords")
            preprocessing_types.append("stopwords")
        for out_column, preprocessing_type in zip(out_columns, preprocessing_types):
            now = datetime.now()
            if "lemmatized" not in out_column:
                columns_to_stem.append(out_column)
            if out_column in out_data_df.columns and out_column not in columns_to_overwrite:
                print(f"\tNot creating {out_column}: already exists", end=" ")
                print(f"If you want to run the preprocessing again, pass {out_column} to --overwrite")
                continue
            elif out_column in out_data_df.columns and out_column in columns_to_overwrite:
                print(f"\t{now} Creating {out_column}: overwriting")
            else:
                print(f"\t{now} Creating {out_column}")
            preprocessed_texts = process_texts(list(out_data_df[column].values),
                                               preprocessing_type=preprocessing_type)
            out_data_df[out_column] = preprocessed_texts
            out_data_df.to_csv(out_data_name, index=None)

if args.stemming:
    out_data_df = pd.read_csv(out_data_name)
    for column in columns_to_stem:
        now = datetime.now()
        out_column = f"{column}_stemmed"
        if out_column in out_data_df.columns and not args.overwrite:
            print(f"\tNot stemming {column}: {out_column} already exists", end=" ")
            print(f"If you want to run the preprocessing again, pass {out_column} to --overwrite")
            continue
        elif out_column in out_data_df.columns and out_column in columns_to_overwrite:
            print(f"\t{now} Stemming {column}: overwriting {out_column}")
        else:
            print(f"\t{now} Stemming {column}")
        preprocessed_texts = process_texts(list(out_data_df[column].values),
                                           preprocessing_type="stemming")
        out_data_df[out_column] = preprocessed_texts
        out_data_df.to_csv(out_data_name, index=None)

print(f"\t{datetime.now()} Done.")
