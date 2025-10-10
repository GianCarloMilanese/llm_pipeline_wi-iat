#!/usr/bin/env python3

import glob
import numpy as np
import os
import pandas as pd
import re
import spacy

from collections import Counter
from functools import partial
from multiprocessing import Pool

from preprocessing_utils import preprocess_tweet


def percent_of_correct_lemmas(original_nlp, processed_nlp):
    processed_tokens = [token.text.lower()
                        for token in processed_nlp
                        if not token.is_punct]

    # Use a counter to account for multiple occurrences of a lemma
    processed_counter = Counter(processed_tokens)

    n_correctly_processed = 0
    total = 0
    for word in original_nlp:
        if word.is_punct:
            continue
        lemma = word.lemma_.lower()
        if processed_counter[lemma] > 0:
            n_correctly_processed += 1
            processed_counter[lemma] -= 1
        total += 1

    if total == 0:
        return np.nan
    return n_correctly_processed/total

def _analyze_file_lemmatization(filename, nlp):
    short_filename = os.path.basename(filename).replace(".csv", "")

    df = pd.read_csv(filename)
    col = "Text_lemmatized"

    texts = [str(text) for text in list(df["Text"])]
    processed_texts = [str(text) for text in list(df[col])]

    texts = [preprocess_tweet(text) for text in texts]
    processed_texts = [preprocess_tweet(text) for text in processed_texts]

    text_nlps = [nlp(text) for text in texts]
    processed_text_nlps = [nlp(text) for text in processed_texts]

    lemma_ratio = np.nanmean([percent_of_correct_lemmas(x, y)
                              for x, y in zip(text_nlps, processed_text_nlps)])

    print(f"Done with {short_filename}")
    return (short_filename, col, lemma_ratio)

def analyze_lemmatization(files, output_file, nlp, n_cpus=7):
    with Pool(n_cpus) as pool:
        output = pool.map(partial(_analyze_file_lemmatization, nlp=nlp), files)

    out_df = pd.DataFrame(output, columns=["filename", "column", "perc_correct_lemmas"])
    out_df = out_df.sort_values(["filename", "column"])
    output_csv = output_file+".csv"
    out_df.to_csv(output_csv, index=False, float_format='%.4f')


if __name__ == "__main__":

    import multiprocessing
    n_cpus = (multiprocessing.cpu_count()//4)+1

    os.makedirs("./results_analysis", exist_ok=True)

    print("\nLEMMATIZATION ANALYSIS")

    files = glob.glob("./datasets_preprocessed/*.csv")

    NLP_EN = spacy.load('en_core_web_sm')
    nonenglish_regex = re.compile("italian|french|spanish|german|portuguese")
    english_files = [file for file in files if not nonenglish_regex.search(file)]
    analyze_lemmatization(english_files, "./results_analysis/lemmatization_statistics",
                          nlp=NLP_EN, n_cpus=n_cpus)

    languages = ["french", "german", "italian", "spanish", "portuguese"]
    abbrs = ["fr", "de", "it", "es", "pt"]
    for lang, abbr in zip(languages, abbrs):
        NLP = spacy.load(f'{abbr}_core_news_sm')
        lang_files = [file for file in files if lang in file]
        analyze_lemmatization(lang_files,
                              f"./results_analysis/lemmatization_statistics_{lang}",
                              nlp=NLP, n_cpus=n_cpus)
