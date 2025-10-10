#!/usr/bin/env python3

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from preprocessing_utils import is_punctuation


def valid_word(word):
    if is_punctuation(word):
        return False
    if word.isnumeric():
        return False
    return True

def get_vocabulary(filenames):
    volcabulary = set()
    lengths = []
    for filename in filenames:
        df = pd.read_csv(filename)
        texts = df["Text"].values
        labels = df["Label"].values
        n = len(texts)
        print(f"{filename} size = {n}")
        print(f"{filename} labels = {len(set(labels))}")
        for text in texts:
            words = word_tokenize(str(text))
            lengths.append(len(words))
            words = [word for word in words if valid_word(word)]
            volcabulary.update(words)
    mean = np.mean(lengths)
    print(f"{filenames} average text lengths: {mean} words")
    return volcabulary


if __name__ == "__main__":
    import glob
    import re

    files = glob.glob("../data/**/*.csv")

    nonenglish_regex = re.compile("italian|french|spanish|german|portuguese")
    english_files = [file for file in files if not nonenglish_regex.search(file)]
    en_vocabulary = get_vocabulary(english_files)
    len_en_vocabulary = len(en_vocabulary)
    print(f"{len_en_vocabulary = }")

    emoji = [file for file in files if "semevalemoji" in file.lower()]
    hate = [file for file in files if "semevalhate" in file.lower()]
    irony = [file for file in files if "semevalirony" in file.lower()]
    sentiment = [file for file in files if "semevalsentiment" in file.lower()]
    offensive = [file for file in files if "semevaloffensive" in file.lower()]

    len_emoji_voc = len(get_vocabulary(emoji))
    len_hate_voc = len(get_vocabulary(hate))
    len_irony_voc = len(get_vocabulary(irony))
    len_sentiment_voc = len(get_vocabulary(sentiment))
    len_offensive_voc = len(get_vocabulary(offensive))

    print(f"{len_emoji_voc = }")
    print(f"{len_hate_voc = }")
    print(f"{len_irony_voc = }")
    print(f"{len_sentiment_voc = }")
    print(f"{len_offensive_voc = }")

    languages = ["french", "german", "italian", "spanish", "portuguese"]
    abbrs = ["fr", "de", "it", "es", "pt"]
    for lang, abbr in zip(languages, abbrs):
        lang_files = [file for file in files if lang in file]
        len_vocabulary = len(get_vocabulary(lang_files))
        print(f"{lang} vocabulary size = {len_vocabulary}")

