#!/usr/bin/env python3
"Text classification on the traditionally preprocessed texts"

import string
import spacy
import os
import datasets
import csv
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

PUNCTUATIONS = string.punctuation
PUNCTUATIONS += '$'+'£'+'&'


def remove_stopwords(text, lang):
   STOPWORDS = stopwords.words(lang)
   if lang =='english':
      STOPWORDS = [x for x in STOPWORDS if x not in ['not']]
   elif lang == 'french':
      STOPWORDS = [x for x in STOPWORDS if x not in ['pas','non']]
   elif lang == 'ferman':
      STOPWORDS = [x for x in STOPWORDS if x not in ['nein','nicht']]
   elif lang == 'italian':
      STOPWORDS = [x for x in STOPWORDS if x not in ['no','non']]
   elif lang == 'spanish':
      STOPWORDS = [x for x in STOPWORDS if x not in ['no']]
   elif lang == 'portuguese':
      STOPWORDS = [x for x in STOPWORDS if x not in ["não", "nem"]]

   return ' '.join([word for word in text.split() if word not in STOPWORDS])


def remove_punctuation(text):
    trans = str.maketrans(dict.fromkeys(PUNCTUATIONS, ' '))
    return text.translate(trans)

def stemSentence(text, stemmer):
    #Token are defined by space between words
    token_words=text.split()
    stem_text=""
    for word in token_words:
        # Stemming
        stem_text=stem_text+' '+stemmer.stem(word)
    # Return string (as input type)
    return stem_text


def LemmSentence(text, lemmatizer):
  load_model = lemmatizer
  try:
   doc = load_model(text)
  except:
      import ipdb
      ipdb.set_trace()
  # Extract the lemma for each token and join
  return " ".join([token.lemma_ for token in doc])



def preprocessing_cleaning(dataset, lang, remove_stopwords_bool, stemming, stemmer, lemmatization, lemmatizer):
   
   dataset['preprocessed'] = dataset['preprocessed'].str.lower()

   if stemming:
      if remove_stopwords_bool:
         dataset['preprocessed']=dataset['preprocessed'].map(lambda x: remove_stopwords(str(x), lang))  

      dataset['preprocessed']=dataset['preprocessed'].map(lambda x: stemSentence(str(x),stemmer))
   
   if lemmatization:
      dataset['preprocessed']=dataset['preprocessed'].map(lambda x: LemmSentence(str(x), lemmatizer))

      if remove_stopwords_bool:
         dataset['preprocessed']=dataset['preprocessed'].map(lambda x: remove_stopwords(str(x), lang))  

   if (remove_stopwords_bool and not stemming and not lemmatization):
      dataset['preprocessed']=dataset['preprocessed'].map(lambda x: remove_stopwords(str(x),lang))  

   return dataset




def ml_algorithm(train,test,column, feat, ngram, ml, results):
   
   tfidf_vect = TfidfVectorizer(max_features=feat, ngram_range=(1,ngram))
   X_train_transformed = tfidf_vect.fit_transform(train[column])

   try:
      label = train['Label']
   except:
      label = train['label']


   if ml == 'Bernoulli':        
      alg = MultinomialNB().fit(X_train_transformed, label)
   elif ml =='DecisionTree':
      alg = DecisionTreeClassifier(random_state=42).fit(X_train_transformed, label)
   elif ml == 'Regression':
      alg = LogisticRegression(random_state=42).fit(X_train_transformed, label)


   X_val_transformed = tfidf_vect.transform(test[column])

   
   y_pred = alg.predict(X_val_transformed)

   try:
      test_label = test['Label']
   except:
      test_label = test['label']

   results.append( {'ngrams': ngram, 'feat': feat, 'column' : column, 'algorithm': alg, 'F1_micro':f1_score(test_label,y_pred,average='micro')}) 
   return results



def save_dataset(results,name):
   field_names = ['ngrams','feat','column','algorithm','F1_macro','F1_micro']
  
   with open('./results/'+name+'.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = field_names) 
    writer.writeheader()
    for el in results:
       writer.writerows(el)


def clean_df(column, train, test):
   train['preprocessed'] = train[column].str.lower()
   test['preprocessed'] = test[column].str.lower()

   train['preprocessed'] = train['preprocessed'].str.replace('\n', ' ')
   test['preprocessed'] = test['preprocessed'].str.replace('\n', ' ')
   
   train['preprocessed']=train['preprocessed'].map(lambda x: remove_punctuation(str(x)))
   test['preprocessed']=test['preprocessed'].map(lambda x: remove_punctuation(str(x)))

   train = train.fillna('')
   test = test.fillna('')

   train = train.sample(frac=1, random_state = 42).reset_index(drop=True)
   test = test.sample(frac=1, random_state = 42).reset_index(drop=True)

   return train, test


directory_df= os.fsencode('../data/train')

language_id = {'italian':'it','spanish':'es','german':'de','french':'fr','portuguese':'pt'}

best_config = {'DecisionTree':{'Stopwords':{'feat':3000,'ngrams':1}, 'Stopwords and Lemmatization':{'feat':3000,'ngrams':1}, 'Lemmatization':{'feat':5000,'ngrams':2},
      'Stemming':{'feat':5000,'ngrams':2}, 'Stopwords and Stemming':{'feat':3000,'ngrams':1}},
      
      'Bernoulli':{'Stopwords':{'feat':5000,'ngrams':2}, 'Stopwords and Lemmatization':{'feat':7000,'ngrams':2}, 'Lemmatization':{'feat':7000,'ngrams':2},
      'Stemming':{'feat':7000,'ngrams':3}, 'Stopwords and Stemming':{'feat':7000,'ngrams':3}},
      
      'Regression':{'Stopwords':{'feat':7000,'ngrams':2}, 'Stopwords and Lemmatization':{'feat':5000,'ngrams':3}, 'Lemmatization':{'feat':5000,'ngrams':2},
      'Stemming':{'feat':7000,'ngrams':3}, 'Stopwords and Stemming':{'feat':5000,'ngrams':2}}}

for file in os.listdir(directory_df):
   filename = os.fsdecode(file)
   
   if ('train' in filename) and ('semeval' in filename):
      
      train = pd.read_csv('../data/train/'+filename)
      test = pd.read_csv('../data/test/'+filename.replace('trainAll', 'test'))

      column='Text'
      train, test = clean_df(column,train,test)


      for type_pre in ['Stemming Lancaster', 'Stemming Porter', 'Stopwords', 'Stopwords and Lemmatization', 'Lemmatization', 'Stemming Lancaster', 'Stemming Snowball','Stopwords and Stemming Porter', 'Stopwords and Stemming Lancaster', 'Stopwords and Stemming Snowball']:
        
        if 'Stopwords' in type_pre:
            rs = True
            ty_re = type_pre
        else:
            rs = False
        if 'Stemming' in type_pre:
            stemming = True
            lemmatization = False
            if 'Porter' in type_pre:
               stemmer = PorterStemmer()
            elif 'Lancaster' in type_pre:
               stemmer = LancasterStemmer()
            elif 'Snowball' in type_pre:
               stemmer = SnowballStemmer('english')
            lemmatizer = None

            ty_re = ' '.join(type_pre.split(' ')[:-1])
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
            ty_re = type_pre

        print('Start Pre-processing:')
        print(type_pre)

        train = preprocessing_cleaning(train, 'english', rs, stemming, stemmer, lemmatization, lemmatizer)
        test = preprocessing_cleaning(test, 'english', rs, stemming, stemmer, lemmatization, lemmatizer)
        results = []
        for alg in ['Bernoulli','DecisionTree','Regression']:
           results.append(ml_algorithm(train,test,'preprocessed',best_config[alg][ty_re]['feat'],best_config[alg][ty_re]['ngrams'],alg,[]))
            
        save_dataset(results, type_pre+'_traditional_approach'+filename.replace('.csv',''))


for lang in ['italian','french','german','spanish','portuguese']:
   
   dataset = datasets.load_dataset('cardiffnlp/tweet_sentiment_multilingual', lang)
   
   train = pd.DataFrame(dataset['train'])
   
   test = pd.DataFrame(dataset['test'])

   column='text'
   train, test = clean_df(column,train,test)


   for type_pre in ['Stopwords', 'Stopwords and Lemmatization', 'Lemmatization', 'Stemming Snowball', 'Stopwords and Stemming Snowball']:
        
        if 'Stopwords' in type_pre:
            rs = True
            ty_re = type_pre
        else:
            rs = False
        if 'Stemming' in type_pre:
            stemming = True
            lemmatization = False
            stemmer = SnowballStemmer(lang)
            lemmatizer = None

            ty_re = ' '.join(type_pre.split(' ')[:-1])
        elif 'Lemmatization' in type_pre:
            stemming = False
            lemmatization = True
            stemmer = None
            lemmatizer = spacy.load(language_id[lang]+'_core_news_sm', disable = ['parser','ner'])
            ty_re = type_pre
        else:
            stemming = False
            lemmatization = False
            stemmer = None
            lemmatizer = None
            ty_re = type_pre

        print('Start Pre-processing:')
        print(type_pre)

        train = preprocessing_cleaning(train, 'english', rs, stemming, stemmer, lemmatization, lemmatizer)
        test = preprocessing_cleaning(test, 'english', rs, stemming, stemmer, lemmatization, lemmatizer)
        results = []
        for alg in ['Bernoulli','DecisionTree','Regression']:
           results.append(ml_algorithm(train,test,'preprocessed',best_config[alg][ty_re]['feat'],best_config[alg][ty_re]['ngrams'],alg,[]))
            
        save_dataset(results, type_pre+'_traditional_approach'+'cardiff_'+lang)
