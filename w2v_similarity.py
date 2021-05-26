import pandas as pd
from gensim.models import Word2Vec
import pytrec_eval
from scipy.spatial import distance
import os


def manhattan_distance(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))


def calculate_sim(path_to_file, w, d, true_dict, metrics, sample_word, sim_name):
    os.chdir(path_to_file)
    df = pd.DataFrame()
    temp_dict = {}
    count = 0
    pytrec_dict = {}
    metric_string = ''
    for item in range(len(metrics)):
        if item != len(metrics) - 1:
            temp = f'\'{metrics[item]}\','
            metric_string = metric_string + temp
        else:
            temp = f'\'{metrics[item]}\''
            metric_string = metric_string + temp
    for window in w:
        for size in d:
            model_name = "w2v_new_model_window-{}_size-{}.mdl".format(window, size)
            column_name = "W2V_model_window-{}_size-{}.mdl".format(window, size)
            first_model = f'{model_name}'
            loaded_wv_first = Word2Vec.load(first_model)
            if sim_name == 'cos':
                sim_score = loaded_wv_first.wv.most_similar(positive=[sample_word])
                if df.empty:
                    df = pd.DataFrame(sim_score, columns=[column_name, 'Similarity'])
                else:
                    df_temp = pd.DataFrame(sim_score, columns=[column_name, 'Similarity'])
                    df = pd.concat([df, df_temp], axis=1)
            elif sim_name == 'man':
                for i in loaded_wv_first.wv.vocab:
                    if i != sample_word:
                        sim_score = manhattan_distance(loaded_wv_first.wv[sample_word], loaded_wv_first.wv[i])
                        temp_dict[i] = sim_score
                man_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda item: item[1])[:10],
                                      columns=[model_name, 'similarity'])
                if df.empty:
                    df = man_df
                else:
                    df = pd.concat([df, man_df], axis=1)
            elif sim_name == 'euc':
                for i in loaded_wv_first.wv.vocab:
                    if i != sample_word:
                        sim_score = distance.euclidean(loaded_wv_first.wv[sample_word], loaded_wv_first.wv[i])
                        temp_dict[i] = sim_score
                euc_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda item: item[1])[:10],
                                      columns=[model_name, 'similarity'])
                if df.empty:
                    df = euc_df
                else:
                    df = pd.concat([df, euc_df], axis=1)

            qrel = {
                sample_word: {true_dict[sample_word]
                              }
                }

            run = {
                sample_word: {
                    df.iloc[0][0]: df.iloc[0][1],
                    df.iloc[1][0]: df.iloc[0][1],
                    df.iloc[2][0]: df.iloc[0][1],
                    df.iloc[3][0]: df.iloc[0][1],
                    df.iloc[4][0]: df.iloc[0][1],
                    df.iloc[5][0]: df.iloc[0][1],
                    df.iloc[6][0]: df.iloc[0][1],
                    df.iloc[7][0]: df.iloc[0][1],
                    df.iloc[8][0]: df.iloc[0][1],
                    df.iloc[9][0]: df.iloc[0][1]
                }
            }

            evaluator = pytrec_eval.RelevanceEvaluator(
                qrel, {metric_string})
            pytrec_dict.update(evaluator.evaluate(run))
            count += 1
    filename = f"W2V_{sim_name}_similarity_{sample_word}.csv"
    savedresult = path_to_file + filename
    df.to_csv(savedresult)

    return pytrec_dict, filename


def evaluate(simpsons_data, w, d, distance_metric, metrics, results_path, word_dict):
    df = pd.DataFrame()
    temp_dict = {}
    pytrec_dict = {}
    count = 0
    metric_string = ''
    for item in range(len(metrics)):
        if item != len(metrics) - 1:
            temp = f'\'{metrics[item]}\','
            metric_string = metric_string + temp
        else:
            temp = f'\'{metrics[item]}\''
            metric_string = metric_string + temp
    for window in w[:1]:
        for size in d[:1]:
            model_name = "w2v_new_model_window-{}_size-{}.mdl".format(window, size)
            column_name = "W2V_model_window-{}_size-{}.mdl".format(window, size)
            first_model = f'{simpsons_data}\\{model_name}'
            loaded_wv_first = Word2Vec.load(first_model)
            for word, val in word_dict.items():
                if word in list(loaded_wv_first.wv.vocab):
                    for sim_name in distance_metric:
                        if sim_name == 'cos':
                            sim_score = loaded_wv_first.wv.most_similar(positive=[word])
                            if df.empty:
                                df = pd.DataFrame(sim_score, columns=[column_name, 'Similarity'])
                            else:
                                df_temp = pd.DataFrame(sim_score, columns=[column_name, 'Similarity'])
                                df = pd.concat([df, df_temp], axis=1)
                        elif sim_name == 'man':
                            for i in loaded_wv_first.wv.vocab:
                                if i != word:
                                    sim_score = manhattan_distance(loaded_wv_first.wv[word], loaded_wv_first.wv[i])
                                    temp_dict[i] = sim_score
                            man_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda item: item[1])[:10],
                                                  columns=[model_name, 'similarity'])
                            if df.empty:
                                df = man_df
                            else:
                                df = pd.concat([df, man_df], axis=1)
                        elif sim_name == 'euc':
                            for i in loaded_wv_first.wv.vocab:
                                if i != word:
                                    sim_score = distance_metric.euclidean(loaded_wv_first.wv[word],
                                                                          loaded_wv_first.wv[i])
                                    temp_dict[i] = sim_score
                            euc_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda item: item[1])[:10],
                                                  columns=[model_name, 'similarity'])
                            if df.empty:
                                df = euc_df
                            else:
                                df = pd.concat([df, euc_df], axis=1)
                    qrel = {
                        word: val
                    }
                    run = {
                        word:
                            {
                                df.iloc[0][0]: df.iloc[0][1],
                                df.iloc[1][0]: df.iloc[0][1],
                                df.iloc[2][0]: df.iloc[0][1],
                                df.iloc[3][0]: df.iloc[0][1],
                                df.iloc[4][0]: df.iloc[0][1],
                                df.iloc[5][0]: df.iloc[0][1],
                                df.iloc[6][0]: df.iloc[0][1],
                                df.iloc[7][0]: df.iloc[0][1],
                                df.iloc[8][0]: df.iloc[0][1],
                                df.iloc[9][0]: df.iloc[0][1]
                            }
                    }

                    evaluator = pytrec_eval.RelevanceEvaluator(
                        qrel, {metric_string})
                    pytrec_dict.update(evaluator.evaluate(run))
                    count += 1
            df.to_csv(results_path)
            return pytrec_dict
