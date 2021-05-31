import re
import pandas as pd
from time import time
import spacy


def clean(doc):
    txt = [token.lemma_ for token in doc if not token.is_stop]
    if len(txt) > 2:
        return ' '.join(txt)


def load_corpus(path_to_file, path_to_save):
    try:
        return pd.read_csv(path_to_save)
    except:
        df = pd.read_csv(path_to_file)
        df = df.dropna().reset_index(drop=True)
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
        brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['spoken_words'])
        t = time()
        txt = [clean(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000)]
        print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
        df_clean = pd.DataFrame({'clean': txt})
        df_clean = df_clean.dropna().drop_duplicates()
        # df_clean.to_csv('/scratch/karan96/w2v_project/output')#what's this Karan!!!!
        df_clean.to_csv(path_to_save)
    return df_clean
