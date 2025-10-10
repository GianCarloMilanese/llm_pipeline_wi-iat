#!/usr/bin/env python3
"Save cardiffnlp tweet multilingual dataset in the right format for llm_preprocessing.py"

import os

from datasets import load_dataset
from tqdm import tqdm

dataset_name = "cardiffnlp/tweet_sentiment_multilingual"
languages = ["french", "german", "italian", "spanish", "portuguese"]

for language in tqdm(languages):
    for split in ["train", "test"]:
            dataset = load_dataset(dataset_name, language, split=split)
            out_folder = f"../data/{split}"
            os.makedirs(out_folder, exist_ok=True)
            df = dataset.to_pandas()
            df.columns =  ["Text", "Label"]
            out_filename = os.path.join(out_folder, f"cardiffnlp_{language}_{split}.csv")
            print(out_filename)
            df.to_csv(out_filename)
