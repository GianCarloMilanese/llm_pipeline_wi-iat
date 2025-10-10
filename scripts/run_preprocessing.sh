#!/usr/bin/env bash

models=(
    google/gemma-2-9b-it
    google/gemma-3-4b-it
    Qwen/Qwen2.5-7B-Instruct
    meta-llama/Llama-3.1-8B-Instruct
    microsoft/Phi-3-mini-4k-instruct
)

config="./config/cardiffnlp_spanish_en-prompt.json"
train_data="../data/train/cardiffnlp_spanish_train.csv"
test_data="../data/test/cardiffnlp_spanish_test.csv"
out_dir="./datasets_preprocessed/"

for model in ${models[@]}
do

    dt=$(date '+%d/%m/%Y %H:%M:%S');
    echo "$dt Processing train"
    python -u ./llm_preprocessing.py $train_data --config $config -m $model -l -w -s -d $out_dir --gpu cuda:1

    dt=$(date '+%d/%m/%Y %H:%M:%S');
    echo "$dt Processing test"
    python -u ./llm_preprocessing.py $test_data --config $config -m $model -l -w -s -d $out_dir --gpu cuda:1

done

config="./config/cardiffnlp_spanish_es-prompt.json"
train_data="../data/train/cardiffnlp_spanish_train.csv"
test_data="../data/test/cardiffnlp_spanish_test.csv"
out_dir="./datasets_preprocessed/"

for model in ${models[@]}
do

    dt=$(date '+%d/%m/%Y %H:%M:%S');
    echo "$dt Processing train"
    python -u ./llm_preprocessing.py $train_data --config $config -m $model -l -w -s -d $out_dir --gpu cuda:1

    dt=$(date '+%d/%m/%Y %H:%M:%S');
    echo "$dt Processing test"
    python -u ./llm_preprocessing.py $test_data --config $config -m $model -l -w -s -d $out_dir --gpu cuda:1

done
