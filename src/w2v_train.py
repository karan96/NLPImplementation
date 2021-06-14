from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import os


def phraser_train(df_clean):
    sent = [row.split() for row in df_clean['clean']]
    phrases = Phrases(sent, min_count=30, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[sent]
    return sentences


def build_model(w, d, cores, sentences, path_to_save):
    model_list = []

    for window in w:
        for size in d:
            model_name = "w2v_w{}_d{}.mdl".format(window, size)
            model_list.append(f'{model_name}')
            if os.path.isfile(f'{path_to_save}{model_name}'): continue #avoid retraining for existing models
            w2v = Word2Vec(min_count=20,
                           iter=100,
                           window=window,
                           size=size,
                           sample=6e-5,
                           alpha=0.03,
                           min_alpha=0.0007,
                           negative=20,
                           workers=cores - 1)

            w2v.build_vocab(sentences, progress_per=10000)
            w2v.train(sentences, total_examples=w2v.corpus_count, epochs=10, report_delay=1)
            w2v.save(f'{path_to_save}{model_name}')

    return model_list
