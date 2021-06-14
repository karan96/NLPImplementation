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
log_path = f'../output/logs/'
full_path = f'../data/raw/'
output_path = f'../output/'
brown_path = f'../output/brown/'
evaluation_charts = f'../evaluation/charts/'
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO,
                    filename=log_path)
evaluation_results = f'../evaluation/results'


# setting up window size range and dimensionality range
cores = multiprocessing.cpu_count()
w = list(range(2, 11))
d = list(range(10, 100, 10)) + list(range(100, 600, 100))

# Pre Processing and Training
distance = ['cos', 'man', 'euc']
metrics = {'map', 'ndcg'}

true_dict = {'homer': {'marge': 10,
                  'sorry': 9,
                  'sweetie': 8,
                  'Married': 7,
                  'far': 6,
                  'father': 5,
                  'wife': 4,
                  'husband': 3,
                  'extra': 2,
                  'ask': 1}}


df_clean = Pre.load_corpus(simpson_file, '../data/preprocessed/simpsons.csv')
sentences = genf.phraser_train(df_clean)
model_list = genf.build_model(w, d, cores, sentences, output_path + 'simpsons/')

pytrec_dict = EW.calculate_sim(output_path + '/simpsons/', model_list, true_dict, metrics, "homer", sim_name='cos')
print(pytrec_dict)
vs.vis(pytrec_dict, evaluation_charts, 'Similarity Comparison for the Word Homer using Cosine Similarity Metric')
pytrec_dict = EW.calculate_sim(output_path + '/simpsons/', model_list, true_dict, metrics, "homer", sim_name='man')
print(pytrec_dict)
vs.vis(pytrec_dict, evaluation_charts, 'Similarity Comparison for the Word Homer using Manhattan Similarity Metric')
pytrec_dict = EW.calculate_sim(output_path + '/simpsons/', model_list, true_dict, metrics, "homer", sim_name='euc')
print(pytrec_dict)
vs.vis(pytrec_dict, evaluation_charts, 'Similarity Comparison for the Word Homer using Euclidean Similarity Metric')

brown_sents = brown.sents(categories='news')
model_list_brown = genf.build_model(w, d, cores, brown_sents, brown_path)
word_dic = GS.process(f'{simFile}')

pytrec_dict = EW.evaluate(brown_path, model_list_brown, distance, metrics, f'{output_path}eval.csv', word_dic)
print(pytrec_dict)
vs.visu(pytrec_dict, evaluation_charts, 'Comparison Against WordSim GoldStandard')
print("The Zero Values indicate that there are no words in the Simpsons dataset")
print(" Whose similar words match with that of WordSim GoldStandard")
