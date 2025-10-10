#!/usr/bin/env python3
"Optimizing hyperparams"

import os
import pandas as pd
import spacy

from nltk.stem.porter import PorterStemmer
from utils_tv import preprocessing_cleaning, ml_algorithm, save_dataset, clean_df

directory = os.fsencode('../data/train/')

for file in os.listdir(directory):
   filename = os.fsdecode(file)
   if ('train' in filename) and ('semeval' in filename and 'Sentiment' in filename) and ('3k' in filename):
      
      train = pd.read_csv('../data/train/'+filename)
      validation = pd.read_csv('../data/'+filename.replace('trainAll_3k','Validation'), nrows = 2000)

      validation = validation.rename(columns={'1':'Text','2':'Label'})

      column='Text'
     
      train, validation = clean_df(column,train,validation)


      for type_pre in ['Stemming Porter', 'Stopwords', 'Stopwords and Lemmatization', 'Lemmatization', 'Stopwords and Stemming Porter']:
        
        if 'Stopwords' in type_pre:
            rs = True
        else:
            rs = False
        if 'Stemming' in type_pre:
            stemming = True
            lemmatization = False
            stemmer = PorterStemmer()
            lemmatizer = None
            
        elif 'Lemmatization' in type_pre:
            stemming = False
            lemmatization = True
            stemmer = None
            lemmatizer = spacy.load('en_core_web_sm', disable = ['parser','ner'])
            ty_re = type_pre
        else:
            stemming = False
            lemmatization = False
            stemmer = None
            lemmatizer = None

        train = preprocessing_cleaning(train, 'english', rs, stemming, stemmer, lemmatization, lemmatizer)
        validation = preprocessing_cleaning(validation, 'english', rs, stemming, stemmer, lemmatization, lemmatizer)
        results = []
        print(type_pre)
        for alg in ['Bernoulli','DecisionTree','Regression']:
            for ngram in [1,2,3]:
                for feat in [500,1000,3000,5000,7000]:
                    results.append(ml_algorithm(train,validation,'preprocessed',feat,ngram,alg,[]))
            
        save_dataset(results, type_pre+'_hyper_parameter_settings'+filename.replace('.csv',''))
