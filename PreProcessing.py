import re
import pandas as pd
from time import time
import spacy

def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)

def first():
    df = pd.read_csv('D:\Desktop\Papers\Data\simpsons_dataset.csv')
    df = df.dropna().reset_index(drop=True)
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed
    brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['spoken_words'])
    t = time()
    txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000)]
    print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    df_clean = pd.DataFrame({'clean': txt})
    df_clean = df_clean.dropna().drop_duplicates()
    return df_clean
