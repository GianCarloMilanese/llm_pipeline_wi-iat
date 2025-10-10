#!/usr/bin/env python3

import glob
import json
import numpy as np
import os
import pandas as pd
import re
import spacy

from collections import Counter
from functools import partial
from multiprocessing import Pool

from preprocessing_utils import (preprocess_tweet, EN_STOPWORDS, IT_STOPWORDS,
                                 ES_STOPWORDS, FR_STOPWORDS, DE_STOPWORDS, PT_STOPWORDS, remove_punctuation)

LANG_TO_STOPWORDS = {
        "english": {"stopwords": EN_STOPWORDS, "nots": {"not"}},
        "french": {"stopwords": FR_STOPWORDS, "nots": {"pas", "non"}},
        "german": {"stopwords": DE_STOPWORDS, "nots": {"nein", "nicht"}},
        "italian": {"stopwords": IT_STOPWORDS, "nots": {"non", "no"}},
        "spanish": {"stopwords": ES_STOPWORDS, "nots": {"no"}},
        "portuguese": {"stopwords": PT_STOPWORDS, "nots": {"não", "nem"}},
        }


def valid_stopword(word, stopwords):
    "Check if a word is a stopword, possibly after removing punctuation"
    stripped_word = word.text.lower().strip()
    no_punct = remove_punctuation(stripped_word)
    if stripped_word in stopwords or no_punct in stopwords:
        return True
    return False

def valid_nonstopword(word, stopwords):
    """Check if a word is not a stopword, possibly after removing punctuation,
    and filtering out empty or punctuation-only strings."""
    if word.is_punct:
        return False
    stripped_word = word.text.lower().strip()
    no_punct = remove_punctuation(stripped_word)
    if no_punct == "":
        return False
    if stripped_word in stopwords:
        return False
    if no_punct in stopwords:
        return False
    return True

def percent_of_removed_stopwords(original_nlp, processed_nlp, language,
                                 exclude_not=False):
    """Return the percentage of stopwords from the original text that are
    removed in the processed text"""
    stopwords = LANG_TO_STOPWORDS[language]["stopwords"]
    nots = LANG_TO_STOPWORDS[language]["nots"]
    if exclude_not:
        stopwords = stopwords-nots

    original_stopwords = [word.text.lower().strip() for word in original_nlp
                          if valid_stopword(word, stopwords)]
    n_original_stopwords = len(original_stopwords)

    if n_original_stopwords == 0:
        return np.nan

    processed_stopwords = [word.text.lower().strip() for word in processed_nlp
                          if valid_stopword(word, stopwords)]

    # Use a counter to account for multiple occurrences of a stopword
    processed_counter = Counter(processed_stopwords)

    n_kept_stopwords = 0
    for sw in original_stopwords:
        if processed_counter[sw] > 0:
            n_kept_stopwords += 1
            processed_counter[sw] -= 1

    return 1 - (n_kept_stopwords/n_original_stopwords)


def percent_of_removed_nonstopwords(original_nlp, processed_nlp, language,
                                    exclude_not=False):
    """Return the percentage of non-stopwords from the original text that are
    removed in the processed text"""
    stopwords = LANG_TO_STOPWORDS[language]["stopwords"]
    nots = LANG_TO_STOPWORDS[language]["nots"]
    if exclude_not:
        stopwords = stopwords-nots

    original_nonstopwords = [word.text.lower().strip() for word in original_nlp
                             if valid_nonstopword(word, stopwords)]

    n_original_nonstopwords = len(original_nonstopwords)
    if n_original_nonstopwords == 0:
        return np.nan

    processed_nonstopwords = [word.text.lower().strip() for word in
                              processed_nlp if valid_nonstopword(word, stopwords)]

    # Use a counter to account for multiple occurrences of a non-stopword
    processed_counter = Counter(processed_nonstopwords)

    n_kept_nonstopwords = 0
    for word in original_nonstopwords:
        if processed_counter[word] > 0:
            n_kept_nonstopwords += 1
            processed_counter[word] -= 1

    return 1 - (n_kept_nonstopwords/n_original_nonstopwords)

def most_common_nonstopwords(original_nlps, processed_nlps, language,
                                    exclude_not=False):
    """Return the most common non-stopwords that are removed from the original
    texts"""
    stopwords = LANG_TO_STOPWORDS[language]["stopwords"]
    nots = LANG_TO_STOPWORDS[language]["nots"]
    if exclude_not:
        stopwords = stopwords-nots

    original_nonstopwords_counter = Counter()
    removed_nonstopwords_counter = Counter()

    for original_nlp, processed_nlp in zip(original_nlps, processed_nlps):

        original_nonstopwords = [word.text.lower().strip()
                                 for word in original_nlp
                                 if valid_nonstopword(word, stopwords)]
        processed_nonstopwords = [word.text.lower().strip()
                                  for word in processed_nlp
                                  if valid_nonstopword(word, stopwords)]
        processed_counter = Counter(processed_nonstopwords)

        for word in original_nonstopwords:
            original_nonstopwords_counter[word] += 1
            if processed_counter[word] > 0:
                processed_counter[word] -= 1
            else:
                removed_nonstopwords_counter[word] += 1

    most_common = dict(removed_nonstopwords_counter.most_common(10))

    nonstopwords_percentage_removed = {}
    for lemma in most_common.keys():
        perc = removed_nonstopwords_counter[lemma]/original_nonstopwords_counter[lemma]
        nonstopwords_percentage_removed[lemma] = perc

    return {"most_common_removed_nonstropwords": most_common,
            "percentage_removed_nonstopwords": nonstopwords_percentage_removed}

def _process_file_stopwords(filename, nlp, language):
    short_filename = os.path.basename(filename).replace(".csv", "")
    exclude_not = bool(re.search(r"Sentiment|Emoji|hate|irony|offensive|cardiffnlp", filename))

    output = {}
    df = pd.read_csv(filename)
    col = "Text_nostopwords"

    texts = [str(text) for text in list(df["Text"])]
    processed_texts = [str(text) for text in list(df[col])]

    texts = [preprocess_tweet(text) for text in texts]
    processed_texts = [preprocess_tweet(text) for text in processed_texts]

    text_nlps = [nlp(text) for text in texts]
    processed_text_nlps = [nlp(text) for text in processed_texts]

    ratios = [percent_of_removed_stopwords(x, y, language=language,
                                           exclude_not=exclude_not)
              for x, y in zip(text_nlps, processed_text_nlps)]
    n_texts_no_stopwords = sum([1 for x in ratios if x is np.nan])
    ratio_removed_stopwords = np.nanmean(ratios)

    ratio_removed_nonstopwords = np.nanmean(
            [percent_of_removed_nonstopwords(x, y, language=language,
                                             exclude_not=exclude_not)
             for x, y in zip(text_nlps, processed_text_nlps)])

    output["row"] = (short_filename, col, n_texts_no_stopwords,
                     ratio_removed_stopwords, ratio_removed_nonstopwords)
    output["most_common"] = (short_filename, most_common_nonstopwords(text_nlps,
                                                                      processed_text_nlps,
                                                                      language,
                                                                      exclude_not))
    print(f"Done with {short_filename}")
    return output

def analyze_stopwords(files, output_file, nlp, language, n_cpus=7):

    with Pool(n_cpus) as pool:
        output = pool.map(partial(_process_file_stopwords, nlp=nlp, language=language), files)

    df_rows = [out["row"] for out in output]
    out_df = pd.DataFrame(df_rows, columns=["filename", "column",
                                            "n_texts_without_stopwords",
                                            "perc_removed_stopwords",
                                            "perc_removed_non-stopwords"])
    out_df = out_df.sort_values(["filename", "column"])

    out_df.to_csv(output_file, index=False, float_format='%.4f')

    json_data = [out["most_common"] for out in output]
    json_data = sorted(json_data, key=lambda x: x[0])
    json_data = dict(json_data)
    output_json = output_file.replace(".csv", "_most_common_nonstopwords.json")
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=2)


if __name__ == "__main__":

    import multiprocessing
    n_cpus = (multiprocessing.cpu_count()//4)+1

    os.makedirs("./results_analysis", exist_ok=True)

    print("\nSTOPWORDS ANALYSIS")

    files = glob.glob("./datasets_preprocessed/*.csv")

    NLP_EN = spacy.load('en_core_web_sm')
    nonenglish_regex = re.compile("italian|french|spanish|german|portuguese")
    english_files = [file for file in files if not nonenglish_regex.search(file)]
    analyze_stopwords(english_files, "./results_analysis/stopwords_statistics.csv",
                      nlp=NLP_EN, language="english", n_cpus=n_cpus)

    languages = ["french", "german", "italian", "spanish", "portuguese"]
    abbrs = ["fr", "de", "it", "es", "pt"]
    for lang, abbr in zip(languages, abbrs):
        NLP = spacy.load(f'{abbr}_core_news_sm')
        lang_files = [file for file in files if lang in file]
        analyze_stopwords(lang_files,
                          f"./results_analysis/stopwords_statistics_{lang}.csv", nlp=NLP,
                          language=lang, n_cpus=n_cpus)
