import nltk
from nltk.corpus import brown
import multiprocessing
from gensim.models import Word2Vec
import GoldStandardProc as GS
import pytrec_eval


nltk.download('brown')

cores = multiprocessing.cpu_count()
w = list(range(2,11))
d = list(range(10, 100, 10)) + list(range(100,600,100))

path = ''
def modeltrain(path_to_file):
    model_name = " "

    for window in w:
        for size in d:
            model_name = "w2v_model_window-{}_size-{}.mdl".format(window, size)
            w2v = Word2Vec(min_count=20,
                           iter=100,
                           window=window,  # just to set as an init value
                           size=size,  # just to set as an init value
                           sample=6e-5,
                           alpha=0.03,
                           min_alpha=0.0007,
                           negative=20,
                           workers=cores - 1)
            w2v.build_vocab(brown.sents(categories='news'), progress_per=10000)

            w2v.train(brown.sents(categories='news'), total_examples=w2v.corpus_count, epochs=50, report_delay=1)
            full name = path_to_file + model_name
            path = path_to_file
            w2v.save(full_name)

word_dic = GS.preprocess(path)

def compare_map(word_dic, path):
    not_found = 0
    total_number, summation = 0, 0
    avg_dict = {}
    avg_score = 0
    missing_words, present_words = list(), list()
    for window in w[:1]:
        for size in d[:1]:

            model_name = "w2v_model_window-{}_size-{}.mdl".format(window, size)
            full_name = path + model_name
            short_name = "W2V_COS_W-{}_S-{}".format(window, size)
            loaded_wv = Word2Vec.load(full_name)
            for word, val in word_dic.items():
                if word in list(loaded_wv.wv.vocab):
                    # present_words.append(word)
                    word_list = loaded_wv.wv.most_similar(positive=[word])
                    # print(word_list)
                    qrel = {
                        word:  # football
                            val  # {'basketball': 7, 'soccer': 9, 'tennis': 7}
                    }

                    run = {
                        word:  # football
                            {
                                word_list[0][0]: word_list[0][1],
                                word_list[1][0]: word_list[0][1],
                                word_list[2][0]: word_list[0][1],
                                word_list[3][0]: word_list[0][1],
                                word_list[4][0]: word_list[0][1],
                                word_list[5][0]: word_list[0][1],
                                word_list[6][0]: word_list[0][1],
                                word_list[7][0]: word_list[0][1],
                                word_list[8][0]: word_list[0][1],
                                word_list[9][0]: word_list[0][1]
                            }
                    }
                    evaluator = pytrec_eval.RelevanceEvaluator(
                        qrel, {'map'})
                    print(evaluator.evaluate(run))
                    print(short_name)
                    for k, v in evaluator.evaluate(run).items():
                        total_number += 1

                        for v1, v2 in v.items():
                            summation += v2

                    # print(total_number)
                    # print(summation)
                    avg_score = summation / total_number
                    # print("Average Score {:.2f}".format(avg_score))
                    avg_dict[short_name] = avg_score
            total_number, summation = 0, 0
    print(avg_dict)

