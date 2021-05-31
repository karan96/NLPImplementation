import os
import nltk
import multiprocessing

import logging  # Setting up the logging to monitor gensim
import preprocessing as Pre
import w2v_train as genf
import w2v_similarity as EW
import gold_standard as GS
import visualization as vs
from nltk.corpus import brown

nltk.download('brown')

# Setting directory path
# This File assumes that the files related to project are stored in Evaluation Folder.

simpson_file = "../data/raw/simpsons_dataset.csv"
simFile = "../evaluation/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt"
simpsons_data = "../output/simpsons/"
log_path = f'../output/logs/' #forward slash works in windows and linux and mac
full_path = f'../data/raw/'
output_path = f'../output/'
brown_path = f'../output/brown/'
evaluation_charts = f'../evaluation/charts/'
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO, filename=log_path)

# setting up window size range and dimensionality range
cores = multiprocessing.cpu_count()
w = list(range(2, 11))
d = list(range(10, 100, 10)) + list(range(100, 600, 100))

# Pre Processing and Training
distance = ['cos', 'man', 'euc']
metrics = {'map', 'ndcg'}

true_dict = {'homer': {'salad': 10,
                  'God': 9,
                  'fat': 8,
                  'Married': 7,
                  'far': 6,
                  'moron': 5,
                  'wife': 4,
                  'medical': 3,
                  'extra': 2,
                  'dammit': 1},#the H should be lower case. It took me an hour to find this bug! Please check your code before send it to me for my review.
            'marge': {
                'wife': 10,
                  'malibu_stacy': 9,
                  'picture': 8,
                  'sweety': 7,
                  'hardly': 6,
                  'carry': 5,
                  'chair': 4,
                  'look': 3,
                  'board': 2,
                  'probably': 1
            }}

##test
w=[2,3]
d=[10, 20]

df_clean = Pre.load_corpus(simpson_file, '../data/preprocessed/simpsons.csv')
sentences = genf.phraser_train(df_clean)
model_list = genf.build_model(w, d, cores, sentences, output_path + '/simpsons/')
pytrec_dict, filename = EW.calculate_sim(output_path + '/simpsons/', model_list, true_dict, metrics, "homer", sim_name='cos')
#vs.visualize(pytrec_dict, evaluation_charts, filename)
pytrec_dict = EW.calculate_sim(output_path + '/simpsons/', model_list, true_dict, metrics, "homer", sim_name='man')
#vs.visualize(pytrec_dict, evaluation_charts, filename)
pytrec_dict = EW.calculate_sim(output_path + '/simpsons/', model_list, true_dict, metrics, "homer", sim_name='euc')
#vs.visualize(pytrec_dict, evaluation_charts, filename)

brown_sents = brown.sents(categories='news')
model_list_brown = genf.build_model(w, d, cores, brown_sents, brown_path)

word_dic = GS.process(f'{simFile}')

# Up to Here
# Print Number of missing words from our Corpus when compared with GoldStandard
GS.missingWords(word_dic, f'{brown_path}w2v_w2_d10.mdl')


pytrec_dict = EW.evaluate(output_path, model_list_brown, distance, metrics, './evaluation/results/', word_dic)
vs.visualize(pytrec_dict, evaluation_charts, 'Comparison Against WordSim GoldStandard')
