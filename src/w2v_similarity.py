import pandas as pd
from gensim.models import Word2Vec
import pytrec_eval
from scipy.spatial import distance


def manhattan_distance(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))


def calculate_sim(path_to_file, models, true_dict, metrics, sample_word, sim_name):
    df_to_csv = pd.DataFrame()
    pytrec_dict = {}
    for model in models:
        model_name = model
        column_name = model
        first_model = model.split('.')[0]
        print(first_model)
        loaded_wv_first = Word2Vec.load(f'{path_to_file}{model_name}')
        if sim_name == 'cos':
            sim_score = loaded_wv_first.wv.most_similar(positive=[sample_word])
            df = pd.DataFrame(sim_score, columns=[column_name, 'Similarity'])
            pytrec_dict_temp, df_temp = pytrec_calc(true_dict, sample_word, df, metrics, pytrec_dict, df_to_csv)
            pytrec_dict[model_name] = pytrec_dict_temp
        elif sim_name == 'man':
            update_dict = {}
            for i in loaded_wv_first.wv.vocab:
                temp_dict = {}
                if i != sample_word:
                    sim_score = manhattan_distance(loaded_wv_first.wv[sample_word], loaded_wv_first.wv[i])
                    temp_dict[i] = sim_score
                    update_dict.update(temp_dict)
            man_df = pd.DataFrame(sorted(update_dict.items(), key=lambda item: item[1])[:10],
                                  columns=[model_name, 'similarity'])
            pytrec_dict_temp, df_temp = pytrec_calc(true_dict, sample_word, man_df, metrics, pytrec_dict, df_to_csv)

            pytrec_dict[model_name] = pytrec_dict_temp
        elif sim_name == 'euc':
            update_dict = {}
            for i in loaded_wv_first.wv.vocab:
                temp_dict = {}
                if i != sample_word:
                    sim_score = distance.euclidean(loaded_wv_first.wv[sample_word], loaded_wv_first.wv[i])
                    temp_dict[i] = sim_score
                    update_dict.update(temp_dict)
            euc_df = pd.DataFrame(sorted(update_dict.items(), key=lambda item: item[1])[:10],
                                  columns=[model_name, 'similarity'])
            pytrec_dict_temp, df_temp = pytrec_calc(true_dict, sample_word, euc_df, metrics, pytrec_dict, df_to_csv)

            pytrec_dict[model_name] = pytrec_dict_temp

    df_to_csv = pd.DataFrame(pytrec_dict.items(), columns=['model_name', 'similarity'])
    print(df_to_csv)
    filename = f"W2V_{sim_name}_similarity_{sample_word}.csv"
    savedresult = path_to_file + filename
    df_to_csv.to_csv(savedresult)
    print("Printing")

    return pytrec_dict


def pytrec_calc(true_dict, sample_word, df, metrics, pytrec_dict, df_to_csv):
    qrel = {
        sample_word: true_dict[sample_word]
        }
    run = {
            sample_word: {
                df.iloc[0][0]: df.iloc[0][1],
                df.iloc[1][0]: df.iloc[1][1],
                df.iloc[2][0]: df.iloc[2][1],
                df.iloc[3][0]: df.iloc[3][1],
                df.iloc[4][0]: df.iloc[4][1],
                df.iloc[5][0]: df.iloc[5][1],
                df.iloc[6][0]: df.iloc[6][1],
                df.iloc[7][0]: df.iloc[7][1],
                df.iloc[8][0]: df.iloc[8][1],
                df.iloc[9][0]: df.iloc[9][1]

            }
    }
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, metrics)
    print(evaluator.evaluate(run))
    pytrec_dict = (evaluator.evaluate(run))
    if df_to_csv.empty:
        df_to_csv = df
    else:
        df_to_csv = pd.concat([df_to_csv, df], axis=1)
    return pytrec_dict, df_to_csv


def evaluate(path_to_file, models, distance_metric, metrics, results_path, word_dict):
    df = pd.DataFrame()
    df_to_csv = pd.DataFrame()
    pytrec_dict = {}
    word_count = 0
    for model in models:
            first_model = model
            loaded_wv_first = Word2Vec.load(f'{path_to_file}{first_model}')
            for word, val in word_dict.items():
                if word in list(loaded_wv_first.wv.vocab):
                    print("Found", word)
                    word_count = word_count + 1
                    for sim_name in distance_metric:
                        print(sim_name)
                        if sim_name == 'cos':
                            sim_score = loaded_wv_first.wv.most_similar(positive=[word])

                            df = pd.DataFrame(sim_score, columns=[first_model, 'Similarity'])

                            df_to_csv = df.copy()
                        elif sim_name == 'man':
                            update_dict = {}
                            for i in loaded_wv_first.wv.vocab:
                                temp_dict = {}
                                if i != word:
                                    sim_score = manhattan_distance(loaded_wv_first.wv[word],
                                                                   loaded_wv_first.wv[i])
                                    temp_dict[i] = sim_score
                                    update_dict.update(temp_dict)
                            man_df = pd.DataFrame(sorted(update_dict.items(), key=lambda item: item[1])[:10],
                                                  columns=[first_model, 'similarity'])

                            df = man_df.copy()
                            df_to_csv.append(df)
                        elif sim_name == 'euc':
                            update_dict = {}
                            for i in loaded_wv_first.wv.vocab:
                                temp_dict = {}
                                if i != word:
                                    sim_score = distance.euclidean(loaded_wv_first.wv[word],
                                                                   loaded_wv_first.wv[i])
                                    temp_dict[i] = sim_score
                                    update_dict.update(temp_dict)
                            euc_df = pd.DataFrame(sorted(update_dict.items(), key=lambda item: item[1])[:10],
                                                  columns=[first_model, 'similarity'])
                            df = euc_df.copy()
                            df_to_csv.append(df)
                        qrel = {
                            word: val
                        }
                        run = {
                            word:
                            {
                                df.iloc[0][0]: df.iloc[0][1],
                                df.iloc[1][0]: df.iloc[1][1],
                                df.iloc[2][0]: df.iloc[2][1],
                                df.iloc[3][0]: df.iloc[3][1],
                                df.iloc[4][0]: df.iloc[4][1],
                                df.iloc[5][0]: df.iloc[5][1],
                                df.iloc[6][0]: df.iloc[6][1],
                                df.iloc[7][0]: df.iloc[7][1],
                                df.iloc[8][0]: df.iloc[8][1],
                                df.iloc[9][0]: df.iloc[9][1]
                            }
                        }
                        evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)
                        pytrec_dict.update(evaluator.evaluate(run))
                print("Total Number of Words Matched with trained vocab of Simpsons Dataset are:", word_count)
    df_to_csv = pd.DataFrame(pytrec_dict.items(), columns=['model_name', 'similarity'])
    df_to_csv.to_csv(results_path)

    return pytrec_dict
