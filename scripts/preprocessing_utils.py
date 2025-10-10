#!/usr/bin/env python3

import string
import re

from nltk.corpus import stopwords

PUNCTUATIONS = string.punctuation + '$£&'
# PUNCTUATIONS = PUNCTUATIONS.replace(r'.', '')
# PUNCTUATIONS = PUNCTUATIONS.replace(r',', '')

EN_STOPWORDS = set(stopwords.words('english'))
ES_STOPWORDS = set(stopwords.words('spanish'))
FR_STOPWORDS = set(stopwords.words('french'))
IT_STOPWORDS = set(stopwords.words('italian'))
DE_STOPWORDS = set(stopwords.words('german'))
PT_STOPWORDS = set(stopwords.words('portuguese'))

# Regex
HASHTAG = re.compile("(?:^|\s)[＃#]{1}(?P<tag>\w+)", re.UNICODE)
CAMELCASE = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
PUNCTUATION_REGEX = re.compile("^[{}]+$".format(re.escape(PUNCTUATIONS)))


def camel_case_split(text):
    matches = CAMELCASE.finditer(text)
    return [m.group(0) for m in matches]

def remove_punctuation(text):
    trans = str.maketrans(dict.fromkeys(PUNCTUATIONS, ' '))
    return text.translate(trans)

def strip_quotes(text):
    return re.sub(r'^("|\')+|("|\')+$', "", str(text))

def is_punctuation(text):
    return bool(PUNCTUATION_REGEX.match(text))

def split_hashtags(text):
    "#IllegalAliens -> Illegal Aliens"
    return HASHTAG.sub(lambda x:" " + " ".join(camel_case_split(x.group("tag"))), text)

def preprocess_tweet(tweet):
    tweet = str(tweet).replace("@user", "user")
    return split_hashtags(tweet)

if __name__ == "__main__":
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
