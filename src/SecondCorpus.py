from gensim.models import Word2Vec
import GoldStandardProc as GS
import pytrec_eval


def compare_map(model_list, path):
    word_dic = GS.preprocess(path)
    not_found = 0
    total_number, summation = 0, 0
    avg_dict = {}
    avg_score = 0
    missing_words, present_words = list(), list()
    for item in model_list:
        loaded_wv = Word2Vec.load(full_name)
        for word, val in word_dic.items():
            if word in list(loaded_wv.wv.vocab):
                word_list = loaded_wv.wv.most_similar(positive=[word])
                qrel = {
                    word: val
                }
                run = {
                    word:
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
                print(item)
                for k, v in evaluator.evaluate(run).items():
                    total_number += 1
                    for v1, v2 in v.items():
                        summation += v2

                    avg_score = summation / total_number
                    avg_dict[item] = avg_score
            total_number, summation = 0, 0
            print("For {} average scores is {}".format(item,avg_dict))

