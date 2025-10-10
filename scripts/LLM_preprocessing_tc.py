#!/usr/bin/env python3
"Text classification on the LLM-preprocessed texts"

import os
import pandas as pd

from utils_tv import ml_algorithm, save_dataset, clean_df

sigle = {'italian':'it','spanish':'es','german':'de','french':'fr','portuguese':'pt'}
best_config = {'DecisionTree':{'Stopwords':{'feat':7000,'ngrams':1}, 'Stopwords and Lemmatization':{'feat':7000,'ngrams':1}, 'Lemmatization':{'feat':1000,'ngrams':2},
      'Stemming':{'feat':500,'ngrams':3}, 'Stopwords and Stemming':{'feat':1000,'ngrams':2}},

      'Bernoulli':{'Stopwords':{'feat':7000,'ngrams':1}, 'Stopwords and Lemmatization':{'feat':1000,'ngrams':2}, 'Lemmatization':{'feat':5000,'ngrams':2},
      'Stemming':{'feat':5000,'ngrams':2}, 'Stopwords and Stemming':{'feat':7000,'ngrams':1}},

      'Regression':{'Stopwords':{'feat':7000,'ngrams':3}, 'Stopwords and Lemmatization':{'feat':5000,'ngrams':3}, 'Lemmatization':{'feat':7000,'ngrams':2},
      'Stemming':{'feat':7000,'ngrams':2}, 'Stopwords and Stemming':{'feat':7000,'ngrams':3}}}

path = "./datasets_preprocessed/"
for file in os.listdir(path):
    filename = os.fsdecode(file)

    if ('train' in filename):
        train= pd.read_csv(path+filename)
        if 'semeval' in filename:
            test = pd.read_csv(path+filename.replace('trainAll','test'))
        elif 'cardiff' in filename:
            test = pd.read_csv(path+filename.replace('train','test'))

        column_list = ['Text_lemmatized', 'Text_nostopwords', 'Text_lemmatized_nostopwords', 'Text_stemmed', 'Text_nostopwords_stemmed']

        for column in column_list:
            print(column)
            train, test = clean_df(column, train, test)

            if column=='Text_lemmatized':
                type_pre = 'Lemmatization'
            elif column=='Text_nostopwords':
                type_pre = 'Stopwords'
            elif column=='Text_lemmatized_nostopwords':
                type_pre = 'Stopwords and Lemmatization'
            elif column == 'Text_stemmed':
                type_pre = 'Stemming'
            elif column == 'Text_nostopwords_stemmed':
                type_pre = 'Stopwords and Stemming'

                results = []

                for alg in list(best_config.keys()):

                    results.append(ml_algorithm(train, test,
                                                'preprocessed',
                                                best_config[alg][type_pre]['feat'],
                                                best_config[alg][type_pre]['ngrams'],
                                                alg,
                                                []))

                    save_dataset(results, 'LLM-preprocessing_'+type_pre+'_'+filename.replace('.csv',''))
