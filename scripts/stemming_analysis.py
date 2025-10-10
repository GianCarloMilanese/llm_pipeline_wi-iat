#!/usr/bin/env python3

import glob
import numpy as np
import os
import pandas as pd
import re

from collections import Counter
from functools import partial
from multiprocessing import Pool
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize

from preprocessing_utils import preprocess_tweet, is_punctuation


def tokenize(sentence):
    return [word.lower()
            for word in word_tokenize(sentence)
            if not is_punctuation(word)]

def percent_of_correct_stems(original_tokens, processed_tokens, stemmer):
    original_stems = [stemmer.stem(word) for word in original_tokens]
    if len(original_stems) == 0:
        return np.nan

    # Use a counter to account for multiple occurrences of a stem
    processed_counter = Counter(processed_tokens)

    n_correctly_processed = 0
    for stem in original_stems:
        if processed_counter[stem] > 0:
            n_correctly_processed += 1
            processed_counter[stem] -= 1

    return n_correctly_processed/len(original_stems)

def percent_of_correct_stems_any(original_tokens, processed_tokens, language):
    if len(original_tokens) == 0:
        return np.nan

    if language == "english":
        porter = PorterStemmer()
        lancaster = LancasterStemmer()
    snowball = SnowballStemmer(language)

    # Use a counter to account for multiple occurrences of a stem
    processed_counter = Counter(processed_tokens)

    n_correctly_processed = 0
    for word in original_tokens:
        if language == "english":
            stem = porter.stem(word)
            if processed_counter[stem] > 0:
                n_correctly_processed += 1
                processed_counter[stem] -= 1
                continue
            stem = lancaster.stem(word)
            if processed_counter[stem] > 0:
                n_correctly_processed += 1
                processed_counter[stem] -= 1
                continue
        stem = snowball.stem(word)
        if processed_counter[stem] > 0:
            n_correctly_processed += 1
            processed_counter[stem] -= 1
            continue
    return n_correctly_processed/len(original_tokens)

def _analyze_file_stemming(filename, language):
    short_filename = os.path.basename(filename).replace(".csv", "")

    df = pd.read_csv(filename)
    col = "Text_stemmed"

    output={"filename": short_filename, "column": col}

    texts = [str(text) for text in list(df["Text"])]
    processed_texts = [str(text) for text in list(df[col])]

    texts = [preprocess_tweet(text) for text in texts]
    processed_texts = [preprocess_tweet(text) for text in processed_texts]

    text_tokens = [tokenize(text) for text in texts]
    processed_text_tokens = [tokenize(text) for text in processed_texts]

    stemmers = [SnowballStemmer(language)]
    stemmer_names = ["Snowball"]
    if language == "english":
        stemmers = [PorterStemmer(), LancasterStemmer(), SnowballStemmer(language)]
        stemmer_names = ["Porter", "Lancaster", "Snowball"]

    for stemmer, stemmer_name in zip(stemmers, stemmer_names):
        stem_ratio = np.nanmean([percent_of_correct_stems(x, y, stemmer=stemmer)
                                  for x, y in zip(text_tokens, processed_text_tokens)])
        output[f"perc_correct_stems_{stemmer_name}"] = stem_ratio

    if language == "english":
        stem_ratio_any = np.nanmean([percent_of_correct_stems_any(x, y, language)
                                  for x, y in zip(text_tokens, processed_text_tokens)])
        output["perc_correct_stems_any"] = stem_ratio_any

    print(f"Done with {short_filename}")
    return output

def analyze_stemming(files, output_file, language, n_cpus=7):
    with Pool(n_cpus) as pool:
        output = pool.map(partial(_analyze_file_stemming, language=language), files)
    out_df = pd.DataFrame.from_records(output,)
    out_df = out_df.sort_values(["filename", "column"])
    out_df.to_csv(output_file, index=False, float_format='%.4f')


if __name__ == "__main__":

    import multiprocessing
    n_cpus = (multiprocessing.cpu_count()//4)+1

    os.makedirs("./results_analysis", exist_ok=True)

    print("\nSTEMMING ANALYSIS")

    files = glob.glob("./datasets_preprocessed/*.csv")

    nonenglish_regex = re.compile("italian|french|spanish|german|portuguese")
    english_files = [file for file in files if not nonenglish_regex.search(file)]
    analyze_stemming(english_files, "./results_analysis/stemming_statistics.csv",
                     language="english", n_cpus=n_cpus)

    for lang in ["french", "german", "italian", "spanish", "portuguese"]:
        lang_files = [file for file in files if lang in file]
        analyze_stemming(lang_files, f"./results_analysis/stemming_statistics_{lang}.csv",
                         language=lang, n_cpus=n_cpus)
