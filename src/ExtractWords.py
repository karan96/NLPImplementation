# please choose a better filename. this file seems to have codes that calculate the similarities.
# so I would suggest to rename it to "w2v_similarity.py"

import pandas as pd
from gensim.models import Word2Vec
import pytrec_eval
import json

# w = list(range(2, 11))
# d = list(range(10, 100, 10)) + list(range(100, 600, 100))

# all the following 3 functions MUST reduce to a single function that accepts a similarity name
# like calculate_sim(path_to_save, w, d, sample_word, sim_name)
# Karan, please consider code reuseability please. never replicate same logic in different locations of pipelines please.
# please have a look at my sample function:

def calculate_sim(path_to_save, w, d, sample_word, sim_name):
    df = pd.DataFrame()
    sim_score = 0
    dict = {}
    for window in w:
        for size in d:
            model_name = "Sample_model_window-{}_size-{}.mdl".format(window, size)
            column_name = "W2V_model_window-{}_size-{}.mdl".format(window, size)
            model_path = ""
            model_path = path_to_save + model_name
            loaded_wv = Word2Vec.load(model_path)
            if sim_name == 'cos':
                sim_score = loaded_wv.wv.most_similar(positive=[sample_word])
                if df.empty:
                    df = pd.DataFrame(sim_score, columns=[column_name, 'Similarity'])
                else:
                    df_temp = pd.DataFrame(sim_score, columns=[column_name, 'Similarity'])
                    df = pd.concat([df, df_temp], axis=1)

            elif sim_name == 'man':
                for i in loaded_wv.wv.vocab:
                    sim_score = distance.euclidean(loaded_wv.wv[sample_word], loaded_wv.wv[i])
                    dict[i] = sim_score
                man_df = pd.DataFrame(sorted(dict.items(), key=lambda item: item[1])[:10], columns=[column_name, 'similarty'])

                if df.empty:
                    df = man_df
                else:
                    df = pd.concat([df, man_df], axis=1)

            elif sim_name == 'euc':
                for i in loaded_wv.wv.vocab:
                    sim_score = distance.euclidean(loaded_wv.wv[sample_word], loaded_wv.wv[i])
                    dict[i] = sim_score
                euc_df = pd.DataFrame(sorted(dict.items(), key=lambda item: item[1])[:10], columns=[column_name, 'similarty'])

                if df.empty:
                    df = euc_df
                else:
                    df = pd.concat([df, euc_df], axis=1)

    fileName = f"W2V_{sim_name}_similarty_{sample_word}.csv"
    savedResult = path_to_save + fileName
    df.to_csv(savedResult)

# def cosine(path_to_save, w, d, sample_word):
#     first_df = pd.DataFrame()
#     first_temp = pd.DataFrame()
#     cos_dict = {}
#     cos_sim = 0
#     for window in w:
#         for size in d:
#             model_name = "Sample_model_window-{}_size-{}.mdl".format(window, size)
#             column_name = "W2V_model_window-{}_size-{}.mdl".format(window, size)
#             model_path = ""
#             model_path = path_to_save + model_name
#             loaded_wv = Word2Vec.load(model_path)
#             cos_sim = loaded_wv.wv.most_similar(positive=[sample_word])
#
#             if first_df.empty:
#                 first_df = pd.DataFrame(cos_sim, columns=[column_name, 'Similarity'])
#             else:
#                 first_temp = pd.DataFrame(cos_sim, columns=[column_name, 'Similarity'])
#                 first_df = pd.concat([first_df, first_temp], axis=1)
#     fileName = f"W2V_cosine_similarty_{sample_word}.csv"
#     savedResult = path_to_file + fileName
#     first_df.to_csv(savedResult)
#
# #there is an error in line44 => what is distance?
# def euc(path_to_save, w, d, sample_word):
#     second_df = pd.DataFrame()
#     default = 0
#     euc_dict = {}
#     for window in w:
#         for size in d:
#             model_name = "Sample_model_window-{}_size-{}.mdl".format(window, size)
#             column_name = "W2V_model_window-{}_size-{}.mdl".format(window, size)
#             loaded_wv = Word2Vec.load(model_name)
#             for i in loaded_wv.wv.vocab:
#                 # why you made an exception for 'marge'
#                 if i != 'marge':
#                     sim = distance.euclidean(loaded_wv.wv[sample_word], loaded_wv.wv[i])
#                     euc_dict[i] = sim
#             euc_df = pd.DataFrame(sorted(euc_dict.items(), key=lambda item: item[1])[:10],
#                                   columns=[column_name, 'similarty'])
#             if second_df.empty:
#                 second_df = euc_df
#             else:
#                 second_df = pd.concat([second_df, euc_df], axis=1)
#     fileName = f"W2V_euclidean_similarty_{sample_word}.csv"
#     savedResult = path_to_save + fileName
#     second_df.to_csv(savedResult)
#
# def manhattan_distance(x,y):
#   return sum(abs(a-b) for a,b in zip(x,y))
#
# def manh(path_to_save, w, d, sample_word):
#     third_df = pd.DataFrame()
#     default = 0
#     euc_dict = {}
#     for window in w:
#         for size in d:
#             model_name = "Sample_model_window-{}_size-{}.mdl".format(window, size)
#             column_name = "W2V_model_window-{}_size-{}.mdl".format(window, size)
#             loaded_wv = Word2Vec.load(model_name)
#             #why you made an exception for 'marge'
#             for i in loaded_wv.wv.vocab:
#                 if i != 'marge':
#                     sim = manhattan_distance(loaded_wv.wv['homer'], loaded_wv.wv[i])
#                     euc_dict[i] = sim
#             euc_df = pd.DataFrame(sorted(euc_dict.items(), key=lambda item: item[1])[:10],
#                                   columns=[column_name, 'similarty'])
#             if third_df.empty:
#                 third_df = euc_df
#             else:
#                 third_df = pd.concat([third_df, euc_df], axis=1)
#     fileName = f"W2V_manhattan_similarty_{sample_word}.csv"
#     savedResult = path_to_save + fileName
#     third_df.to_csv(savedResult)
#

# we don't need the following functions

# def map_cos():
#     for window in w:
#         for size in d:
#             model_name = "Sample_model_window-{}_size-{}.mdl".format(window, size)
#             short_name = "W2V_COS_W-{}_S-{}".format(window, size)
#             loaded_wv = Word2Vec.load(model_name)
#             word_list = loaded_wv.wv.most_similar(positive=["homer"])
#             qrel = {
#                 short_name: {
#                     'salad': 10,
#                     'God': 9,
#                     'fat': 8,
#                     'Married': 7,
#                     'far': 6,
#                     'moron': 5,
#                     'wife': 4,
#                     'medical': 3,
#                     'extra': 2,
#                     'dammit': 1
#                 }
#             }
#
#             run = {
#                 short_name: {
#                     word_list[0][0]: word_list[0][1],
#                     word_list[1][0]: word_list[0][1],
#                     word_list[2][0]: word_list[0][1],
#                     word_list[3][0]: word_list[0][1],
#                     word_list[4][0]: word_list[0][1],
#                     word_list[5][0]: word_list[0][1],
#                     word_list[6][0]: word_list[0][1],
#                     word_list[7][0]: word_list[0][1],
#                     word_list[8][0]: word_list[0][1],
#                     word_list[9][0]: word_list[0][1]
#                 }
#             }
#             evaluator = pytrec_eval.RelevanceEvaluator(
#                 qrel, {'map'})
#             # print(model_name)
#             # print(evaluator.evaluate(run))
#             # print("*****************")
#             dictionary.update(evaluator.evaluate(run))
#             count += 1
#     new_dict = {}
#     for k, v in dictionary.items():
#         # print(k,v)
#         for k2, v2 in v.items():
#             if v2 > 0.05:
#                 new_dict[k] = v
#     new_dictionary = {}
#     new_dictionary = {str((k,k1)):v1 for k, v in new_dict.items() for k1,v1 in v.items()}
#     plt.figure(figsize=(20, 10))
#     plt.bar(new_dictionary.keys(), new_dictionary.values(), width=.5, color='b')
#     plt.xticks(rotation=90, fontsize='small')
#     plt.show()
#
# def map_euc():
#     euc_dict = {}
#     euc_dictionary = {}
#     count = 0
#     for window in w:
#         for size in d:
#             model_name = "Sample_model_window-{}_size-{}.mdl".format(window, size)
#             short_name = "W2V_EUC_W{}_S{}".format(window, size)
#             loaded_wv = Word2Vec.load(model_name)
#             for i in loaded_wv.wv.vocab:
#                 if i != 'homer':
#                     sim = distance.euclidean(loaded_wv.wv['homer'], loaded_wv.wv[i])
#                     euc_dict[i] = sim
#             euc_df = pd.DataFrame(sorted(euc_dict.items(), key=lambda item: item[1])[:10],
#                                   columns=[short_name, 'similarty'])
#             qrel = {
#                 short_name: {
#                     'salad': 10,
#                     'God': 9,
#                     'fat': 8,
#                     'Married': 7,
#                     'far': 6,
#                     'moron': 5,
#                     'wife': 4,
#                     'medical': 3,
#                     'extra': 2,
#                     'dammit': 1
#                 }
#             }
#
#             run = {
#                 short_name: {
#                     euc_df.iloc[0][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[1][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[2][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[3][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[4][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[5][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[6][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[7][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[8][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[9][0]: euc_df.iloc[0][1]
#                 }
#             }
#             evaluator = pytrec_eval.RelevanceEvaluator(
#                 qrel, {'map'})
#             # print(model_name)
#             # print(evaluator.evaluate(run))
#             # print("*****************")
#             euc_dictionary.update(evaluator.evaluate(run))
#             count += 1
#     pount = 0
#     euc_final_dict = {}
#     for k, v in euc_dictionary.items():
#         # print(k,v)
#         for k2, v2 in v.items():
#             if v2 > 0.05:
#                 # print(k2)
#                 euc_final_dict[k] = v
#                 # print(v2)
#                 pount += 1
#     onemore_dict = {}
#     onemore_dict = {str((k, k1)): v1 for k, v in euc_final_dict.items() for k1, v1 in v.items()}
#     plt.figure(figsize=(20, 10))
#     plt.bar(onemore_dict.keys(), onemore_dict.values(), width=.5, color='b')
#     plt.xticks(rotation=90, fontsize='small')
#     plt.show()
#
# def map_man():
#     man_dict = {}
#     man_dictionary = {}
#     for window in w[:5]:
#         for size in d[:5]:
#             model_name = "Sample_model_window-{}_size-{}.mdl".format(window, size)
#             par_name = "W2V_MAN_W{}_S{}".format(window, size)
#             loaded_wv = Word2Vec.load(model_name)
#             for i in loaded_wv.wv.vocab:
#                 if i != 'homer':
#                     sim = manhattan_distance(loaded_wv.wv['homer'], loaded_wv.wv[i])
#                     man_dict[i] = sim
#             man_df = pd.DataFrame(sorted(man_dict.items(), key=lambda item: item[1])[:10],
#                                   columns=[par_name, 'similarty'])
#             qrel = {
#                 par_name: {
#                     'salad': 10,
#                     'God': 9,
#                     'fat': 8,
#                     'Married': 7,
#                     'far': 6,
#                     'moron': 5,
#                     'wife': 4,
#                     'medical': 3,
#                     'extra': 2,
#                     'dammit': 1
#                 }
#             }
#
#             run = {
#                 par_name: {
#                     man_df.iloc[0][0]: man_df.iloc[0][1],
#                     man_df.iloc[1][0]: man_df.iloc[0][1],
#                     man_df.iloc[2][0]: man_df.iloc[0][1],
#                     man_df.iloc[3][0]: man_df.iloc[0][1],
#                     man_df.iloc[4][0]: man_df.iloc[0][1],
#                     man_df.iloc[5][0]: man_df.iloc[0][1],
#                     man_df.iloc[6][0]: man_df.iloc[0][1],
#                     man_df.iloc[7][0]: man_df.iloc[0][1],
#                     man_df.iloc[8][0]: man_df.iloc[0][1],
#                     man_df.iloc[9][0]: man_df.iloc[0][1]
#                 }
#             }
#             evaluator = pytrec_eval.RelevanceEvaluator(
#                 qrel, {'map'})
#             # print(model_name)
#             # print(evaluator.evaluate(run))
#             # print("*****************")
#             man_dictionary.update(evaluator.evaluate(run))
#     man_final_dict = {}
#     for k, v in man_dictionary.items():
#         for k2, v2 in v.items():
#             if v2 > 0.01:
#                 man_final_dict[k] = v
#     man_more_dict = {}
#     man_more_dict = {str((k, k1)): v1 for k, v in man_final_dict.items() for k1, v1 in v.items()}
#     print(man_more_dict)
#     plt.figure(figsize=(20, 10))
#     plt.bar(man_more_dict.keys(), man_more_dict.values(), width=.5, color='b')
#     plt.xticks(rotation=90, fontsize='small')
#     plt.show()
#
# def ndcg_cos():
#     for window in w:
#         for size in d:
#             model_name = "Sample_model_window-{}_size-{}.mdl".format(window, size)
#             short_name = "W2V_COS_W-{}_S-{}".format(window, size)
#             loaded_wv = Word2Vec.load(model_name)
#             word_list = loaded_wv.wv.most_similar(positive=["homer"])
#             qrel = {
#                 short_name: {
#                     'salad': 10,
#                     'God': 9,
#                     'fat': 8,
#                     'Married': 7,
#                     'far': 6,
#                     'moron': 5,
#                     'wife': 4,
#                     'medical': 3,
#                     'extra': 2,
#                     'dammit': 1
#                 }
#             }
#
#             run = {
#                 short_name: {
#                     word_list[0][0]: word_list[0][1],
#                     word_list[1][0]: word_list[0][1],
#                     word_list[2][0]: word_list[0][1],
#                     word_list[3][0]: word_list[0][1],
#                     word_list[4][0]: word_list[0][1],
#                     word_list[5][0]: word_list[0][1],
#                     word_list[6][0]: word_list[0][1],
#                     word_list[7][0]: word_list[0][1],
#                     word_list[8][0]: word_list[0][1],
#                     word_list[9][0]: word_list[0][1]
#                 }
#             }
#             evaluator = pytrec_eval.RelevanceEvaluator(
#                 qrel, {'ndcg'})
#             # print(model_name)
#             # print(evaluator.evaluate(run))
#             # print("*****************")
#             dictionary.update(evaluator.evaluate(run))
#             count += 1
#     new_dict = {}
#     for k, v in dictionary.items():
#         # print(k,v)
#         for k2, v2 in v.items():
#             if v2 > 0.1:
#                 # print(k2)
#                 new_dict[k] = v
#     new_dictionary = {}
#     new_dictionary = {str((k, k1)): v1 for k, v in new_dict.items() for k1, v1 in v.items()}
#     plt.figure(figsize=(20, 10))
#     plt.bar(new_dictionary.keys(), new_dictionary.values(), width=.5, color='b')
#     plt.xticks(rotation=90, fontsize='small')
#     plt.show()
#
# def ndcg_euc():
#     euc_dict = {}
#     euc_dictionary = {}
#     count = 0
#     for window in w:
#         for size in d:
#             model_name = "Sample_model_window-{}_size-{}.mdl".format(window, size)
#             short_name = "W2V_EUC_W{}_S{}".format(window, size)
#             loaded_wv = Word2Vec.load(model_name)
#             for i in loaded_wv.wv.vocab:
#                 if i != 'homer':
#                     sim = distance.euclidean(loaded_wv.wv['homer'], loaded_wv.wv[i])
#                     euc_dict[i] = sim
#             euc_df = pd.DataFrame(sorted(euc_dict.items(), key=lambda item: item[1])[:10],
#                                   columns=[short_name, 'similarty'])
#             qrel = {
#                 short_name: {
#                     'salad': 10,
#                     'God': 9,
#                     'fat': 8,
#                     'Married': 7,
#                     'far': 6,
#                     'moron': 5,
#                     'wife': 4,
#                     'medical': 3,
#                     'extra': 2,
#                     'dammit': 1
#                 }
#             }
#
#             run = {
#                 short_name: {
#                     euc_df.iloc[0][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[1][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[2][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[3][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[4][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[5][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[6][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[7][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[8][0]: euc_df.iloc[0][1],
#                     euc_df.iloc[9][0]: euc_df.iloc[0][1]
#                 }
#             }
#             evaluator = pytrec_eval.RelevanceEvaluator(
#                 qrel, {'ndcg'})
#             # print(model_name)
#             # print(evaluator.evaluate(run))
#             # print("*****************")
#             euc_dictionary.update(evaluator.evaluate(run))
#             count += 1
#     euc_final_dict = {}
#     for k, v in euc_dictionary.items():
#         for k2, v2 in v.items():
#             if v2 > 0.05:
#                 euc_final_dict[k] = v
#     onemore_dict = {}
#     onemore_dict = {str((k, k1)): v1 for k, v in euc_final_dict.items() for k1, v1 in v.items()}
#     plt.figure(figsize=(20, 10))
#     plt.bar(onemore_dict.keys(), onemore_dict.values(), width=.5, color='b')
#     plt.xticks(rotation=90, fontsize='small')
#     plt.show()
#
# def ndcg_manh():
#     man_dict = {}
#     man_dictionary = {}
#     for window in w[:5]:
#         for size in d[:5]:
#             model_name = "Sample_model_window-{}_size-{}.mdl".format(window, size)
#             par_name = "W2V_MAN_W{}_S{}".format(window, size)
#             loaded_wv = Word2Vec.load(model_name)
#             for i in loaded_wv.wv.vocab:
#                 if i != 'homer':
#                     sim = manhattan_distance(loaded_wv.wv['homer'], loaded_wv.wv[i])
#                     man_dict[i] = sim
#             man_df = pd.DataFrame(sorted(man_dict.items(), key=lambda item: item[1])[:10],
#                                   columns=[par_name, 'similarty'])
#             qrel = {
#                 par_name: {
#                     'salad': 10,
#                     'God': 9,
#                     'fat': 8,
#                     'Married': 7,
#                     'far': 6,
#                     'moron': 5,
#                     'wife': 4,
#                     'medical': 3,
#                     'extra': 2,
#                     'dammit': 1
#                 }
#             }
#
#             run = {
#                 par_name: {
#                     man_df.iloc[0][0]: man_df.iloc[0][1],
#                     man_df.iloc[1][0]: man_df.iloc[0][1],
#                     man_df.iloc[2][0]: man_df.iloc[0][1],
#                     man_df.iloc[3][0]: man_df.iloc[0][1],
#                     man_df.iloc[4][0]: man_df.iloc[0][1],
#                     man_df.iloc[5][0]: man_df.iloc[0][1],
#                     man_df.iloc[6][0]: man_df.iloc[0][1],
#                     man_df.iloc[7][0]: man_df.iloc[0][1],
#                     man_df.iloc[8][0]: man_df.iloc[0][1],
#                     man_df.iloc[9][0]: man_df.iloc[0][1]
#                 }
#             }
#             evaluator = pytrec_eval.RelevanceEvaluator(
#                 qrel, {'ndcg'})
#             man_dictionary.update(evaluator.evaluate(run))
#     man_final_dict = {}
#     for k, v in man_dictionary.items():
#         for k2, v2 in v.items():
#             if v2 > 0.01:
#                 man_final_dict[k] = v
#     man_more_dict = {}
#     man_more_dict = {str((k, k1)): v1 for k, v in man_final_dict.items() for k1, v1 in v.items()}
#     print(man_more_dict)
#     plt.figure(figsize=(20, 10))
#     plt.bar(man_more_dict.keys(), man_more_dict.values(), width=.5, color='b')
#     plt.xticks(rotation=90, fontsize='small')
#     plt.show()