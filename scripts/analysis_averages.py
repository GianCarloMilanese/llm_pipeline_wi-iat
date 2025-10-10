#!/usr/bin/env python3
"Compute average results of preprocessing quality for each model and language"

import pandas as pd

def main_english(file, col):
    get_model = lambda filename: filename.split("_")[-1]
    upper = lambda x: x[0].upper()+x[1:]
    df = pd.read_csv(file)
    df["model"] = df.filename.apply(get_model).apply(upper)
    df.drop("filename", axis=1, inplace=True)
    means = df.groupby(["model", "column"]).mean()*100
    print(means[col].round(2).sort_index())

def main_multilanguage(file, col):
    get_model = lambda filename: filename.split("_")[-3]
    get_prompt = lambda filename: filename.split("_")[-2]
    upper = lambda x: x[0].upper()+x[1:]
    df = pd.read_csv(file)
    df["model"] = df.filename.apply(get_model).apply(upper)
    df["prompt"] = df.filename.apply(get_prompt)
    df.drop("filename", axis=1, inplace=True)
    means = df.groupby(["model", "prompt", "column"]).mean()[col]*100
    pivot = means.reset_index().pivot(columns="prompt", values=col, index="model").round(2)
    columns = list(pivot.columns)
    # Sort columns so that "en-prompt" always appears on the left
    columns.remove("en-prompt")
    columns = ["en-prompt"]+columns
    print(pivot[columns].sort_index())

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lemmatization",
                        help="print lemmatization results averages",
                        action="store_true")
    parser.add_argument("-w", "--stopwords",
                        help="print stopword removal results averages",
                        action="store_true")
    parser.add_argument("-n", "--nonstopwords",
                        help="print non-stopword removal results averages",
                        action="store_true")
    parser.add_argument("-s", "--stemming",
                        help="print stemming results averages",
                        action="store_true")
    args = parser.parse_args()

    stemming = args.stemming
    lemmatization = args.lemmatization
    stopwords = args.stopwords
    nonstopwords = args.nonstopwords

    print("English", end=" ")
    if lemmatization:
        print("lemmatization")
        main_english("./results_analysis/lemmatization_statistics.csv",
                     "perc_correct_lemmas")
    if stopwords:
        print("stopwords")
        main_english("./results_analysis/stopwords_statistics.csv",
                     "perc_removed_stopwords")
    if nonstopwords:
        print("nonstopwords")
        main_english("./results_analysis/stopwords_statistics.csv",
                     "perc_removed_non-stopwords")
    if stemming:
        print("stemming")
        print("Porter")
        main_english("./results_analysis/stemming_statistics.csv",
                     "perc_correct_stems_Porter")
        print("Lancaster")
        main_english("./results_analysis/stemming_statistics.csv",
                     "perc_correct_stems_Lancaster")
        print("Snowball")
        main_english("./results_analysis/stemming_statistics.csv",
                     "perc_correct_stems_Snowball")
        print("Any")
        main_english("./results_analysis/stemming_statistics.csv",
                     "perc_correct_stems_any")

    for lang in ["French", "German", "Italian", "Spanish", "Portuguese"]:
        print(f"\n{lang}", end=" ")
        lang = lang.lower()
        if lemmatization:
            print("lemmatization")
            main_multilanguage(f"./results_analysis/lemmatization_statistics_{lang}.csv",
                               "perc_correct_lemmas")
        if stopwords:
            print("stopwords")
            main_multilanguage(f"./results_analysis/stopwords_statistics_{lang}.csv",
                               "perc_removed_stopwords")
        if nonstopwords:
            print("nonstopwords")
            main_multilanguage(f"./results_analysis/stopwords_statistics_{lang}.csv",
                               "perc_removed_non-stopwords")
        if stemming:
            print("stemming")
            main_multilanguage(f"./results_analysis/stemming_statistics_{lang}.csv",
                               "perc_correct_stems_Snowball")
