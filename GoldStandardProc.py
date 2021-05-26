import pandas as pd
from gensim.models import Word2Vec


def process(path_to_file):
    word_dic = {}
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

            for i in range(len(copy_df)):
                temp_dic[copy_df['Pair_2'][i]] = copy_df['Sim_Score'][i]
        word_dic[word] = temp_dic
    for key, value in word_dic.items():
        for x, y in value.items():
            word_dic[key][x] = round(y)
    return word_dic


def missingWords(word_dic, full_name):
    not_found = 0
    loaded_wv = Word2Vec.load(full_name)
    for word, val in word_dic.items():
        if word not in list(loaded_wv.wv.vocab):
            not_found += 1
            print("Number of Gold Truth Words Missing from Corpus are {}".format(not_found))
            print("Words available to match {}".format(len(word_dic) - not_found))
            not_found = 0
