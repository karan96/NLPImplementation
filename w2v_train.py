<<<<<<< HEAD:src/GensimFunc.py
#7. I would choose a better name for this file. I know that you use Gensim lib but you have to choose a name that describe the role of this step in your pipeline.
#e.g., w2v.py or w2v_train.py
#also Func is usually refer to an anonymous function (pointer to function) and has a special meaning in programming languages

from gensim.models.phrases import Phrases, Phraser
import multiprocessing
from gensim.models import Word2Vec


def Phraser(df_clean):
    sent = [row.split() for row in df_clean['clean']]
    phrases = Phrases(sent, min_count=30, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[sent]
    return sentences


# cores = multiprocessing.cpu_count()
# w = list(range(2, 11))
# d = list(range(10, 100, 10)) + list(range(100, 600, 100))


def build_model(sentences, w, d, cores, path_to_save):
    model_list = []
    for window in w:
        for size in d:
            model_name = ""
            full_name = ""
            model_name = "w2v_model_window-{}_size-{}.mdl".format(window, size)
            w2v = Word2Vec(min_count=20,
                           iter=100,
                           window=window,
                           size=size,
                           sample=6e-5,
                           alpha=0.03,
                           min_alpha=0.0007,
                           negative=20,
                           workers=cores - 1)
            full_name = path_to_save + model_name
            w2v.build_vocab(sentences, progress_per=10000)
            w2v.train(sentences, total_examples=w2v.corpus_count, epochs=2, report_delay=1)
            w2v.save(full_name)
            model_list.append(full_name)
    return model_list
=======
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec


def Phraser(df_clean):
    sent = [row.split() for row in df_clean['clean']]
    phrases = Phrases(sent, min_count=30, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[sent]
    return sentences


def build_model(path_to_file, w, d, cores, sentences, eval_path, new_corpus=0):
    model_list = []
    for window in w:
        for size in d:
            model_name = ""
            full_name = ""
            model_name = "w2v_model_window-{}_size-{}.mdl".format(window, size)
            w2v = Word2Vec(min_count=20,
                           iter=100,
                           window=window,
                           size=size,
                           sample=6e-5,
                           alpha=0.03,
                           min_alpha=0.0007,
                           negative=20,
                           workers=cores - 1)
            full_name = os.path.join(eval_path, model_name)

            if new_corpus == 1:
                model_name = "w2v_new_model_window-{}_size-{}.mdl".format(window, size)
                w2v.build_vocab(brown.sents(categories='news'), progress_per=10000)
                w2v.train(brown.sents(categories='news'), total_examples=w2v.corpus_count, epochs=50, report_delay=1)
                full_name = path_to_file + model_name
                w2v.save(full_name)
            else:
                w2v.build_vocab(sentences, progress_per=10000)
                w2v.train(sentences, total_examples=w2v.corpus_count, epochs=50, report_delay=1)
                w2v.save(full_name)
                model_list.append(full_name)
    return model_list
>>>>>>> e6e9609f23223a8b510da5c8e31b55d2e783666a:w2v_train.py
