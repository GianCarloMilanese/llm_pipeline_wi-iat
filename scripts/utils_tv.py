#!/usr/bin/env python3

import csv
import string

from nltk.corpus import stopwords
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
   
   #dataset = dataset[['preprocessed', 'Label']]
   
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
   field_names = ['ngrams','feat','column','algorithm','F1_micro']
  
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
