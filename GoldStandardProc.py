import pandas as pd
from gensim.models import Word2Vec
import pytrec_eval


def preprocess(path_to_file):
    word_dic = {}
    temp_dic = {}
    file_path = path_to_file + 'wordsim_similarity_goldstandard.txt'
    wordsim_df = pd.read_csv(file_path, sep='\t', header=None)

    wordsim_df.columns = ['Pair_1', 'Pair_2', 'Sim_Score']

    clean_wordsim = wordsim_df.loc[wordsim_df['Pair_1'] != wordsim_df['Animal_2']]
    clean_wordsim.reset_index().drop('index', axis=1, inplace=True)

    empty_lists = 0
    for word in clean_wordsim['Pair_1'].unique().tolist():
        one_df = clean_wordsim.loc[clean_wordsim['Pair_1'] == word]
        one_df.sort_values(by='Sim_Score', ascending=False, inplace=True)
        one_df = one_df.reset_index().drop('index', axis=1)
        count = 0
        while len(one_df) < 10:
            if count == len(one_df):
                break
            second_df = clean_wordsim.loc[clean_wordsim['Pair_1'] == one_df['Pair_2'][count]]
            if second_df.empty:
                count += 1
                continue
            else:
                print("Progressing....")
                second_df.replace(to_replace=second_df.iloc[0][0], value='tiger', inplace=True)
                second_df.sort_values(by='Sim_Score', ascending=False, inplace=True)
                second_df = second_df.reset_index().drop('index', axis=1)
                extrar = second_df.loc[~second_df['Pair_2'].isin(one_df['Pair_2']), :]
                if (len(one_df) + len(extrar)) < 10:
                    one_df = pd.concat([one_df, extrar], axis=0)
                elif (len(one_df) + len(extrar)) > 10:
                    diff = 10 - len(one_df)
                    one_df = pd.concat([one_df, extrar[:diff]], axis=0)
            one_df.sort_values(by='Sim_Score', ascending=False, inplace=True)
            one_df = one_df.reset_index().drop('index', axis=1)
            print(len(one_df))
            count += 1
        temp_dic = dict()
        if one_df[one_df['Sim_Score'] > 5.00].empty:
            empty_lists += 1
            continue
        else:
            copy_df = one_df[one_df['Sim_Score'] > 5.00]
            # print(copy_df)
            for i in range(len(copy_df)):
                temp_dic[copy_df['Pair_2'][i]] = copy_df['Sim_Score'][i]
        word_dic[word] = temp_dic
    for key, value in word_dic.items():
        for x, y in value.items():
            word_dic[key][x] = round(y)
    return word_dic


def missingWords(word_dic, full_name):
    not_found = 0
    missing_words, present_words = list(), list()
    loaded_wv = Word2Vec.load(full_name)
    for word, val in word_dic.items():
        if word not in list(loaded_wv.wv.vocab):
            not_found += 1
            print("Number of Gold Truth Words Missing from Corpus are {}".format(short_name, not_found))
            print("Words available to match {}".format(len(word_dic) - not_found))
            not_found = 0

def goldstand_map(path_to_file):
    total_number, summation = 0, 0
    avg_dict = {}
    avg_score = 0
    missing_words, present_words = list(), list()
    for window in w:
        for size in d:
            full_name = path_to_file + 'w2v_model_window-{}_size-{}.mdl'.format(window, size)
            short_name = "w2v_model_window-{}_size-{}.mdl".format(window, size)
            loaded_wv = Word2Vec.load(full_name)
            for word, val in word_dic.items():
                if word in list(loaded_wv.wv.vocab):
                    word_list = loaded_wv.wv.most_similar(positive=[word])
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

                    for k, v in evaluator.evaluate(run).items():
                        total_number += 1

                        for v1, v2 in v.items():
                            summation += v2

                    avg_score = summation / total_number
                    avg_dict[short_name] = avg_score
            total_number, summation = 0, 0
    avg_dict.sort(by='values')

    return avg_dict




