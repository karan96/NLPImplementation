import os
import nltk
import multiprocessing

import logging  # Setting up the logging to monitor gensim
import PreProcessing as Pre
import w2v_train as genf
import w2v_similarity as EW
import GoldStandardProc as GS
import visualization as vs
from nltk.corpus import brown

nltk.download('brown')

# Setting directory path
# This File assumes that the files related to project are stored in Evaluation Folder.

data_file = "simpsons_dataset.csv"
simFile = "wordsim_similarity_goldstandard.txt"
simpsons_data = "./output/simpsons/"

# setting up window size range and dimensionality range

cores = multiprocessing.cpu_count()
w = list(range(2, 11))
d = list(range(10, 100, 10)) + list(range(100, 600, 100))

# Setting up Directory Paths
dir_name = os.path.dirname(__file__)

os.chdir(dir_name)

cwd = os.getcwd()
eval_path = cwd

log_path = f'{eval_path}\\output\\logs\\'
full_path = f'{eval_path}\\data\\raw\\'
output_path = f'{eval_path}\\output\\'
brown_path = f'{eval_path}\\output\\brown\\'
evaluation_charts = f'{eval_path}\\evaluation\\charts\\'

if os.path.exists(evaluation_charts):
    os.chdir(evaluation_charts)
else:
    os.makedirs(evaluation_charts)

if os.path.exists(evaluation_charts):
    os.chdir(evaluation_charts)
else:
    os.makedirs(log_path)

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO,
                    filename=log_path)


def find(simfile, file_path):
    for root, dirs, files in os.walk(file_path):
        if simfile in files:
            print("File Found")
            exit()
        else:
            print("WordSim Similarity Goldstandard file not found.")
            print(
                "Please download it from http://alfonseca.org/pubs/ws353simrel.tar.gz.")
            print("Unzip wordsim_similarity_goldstandard into the evaluation folder ")
            print("Exiting Program")
            exit()


if os.path.exists(full_path):
    os.chdir(full_path)
else:
    os.makedirs(full_path)

if os.path.exists(brown_path):
    os.chdir(log_path)
else:
    os.makedirs(brown_path)

if os.path.exists('evaluation'):
    print("Path Exists Looking for Wordsim Similarity File")
    sim_file = os.path.join(eval_path, 'evaluation')
    print(sim_file)
    find(simFile, sim_file)
else:
    os.makedirs('evaluation')
    sim_file = os.path.join(eval_path, 'evaluation')
    print(sim_file)
    find(simFile, sim_file)

find(simFile, full_path)

path_to_file = os.path.join(full_path, data_file)

# Pre Processing and Training
distance = ['cos', 'man', 'euc']
metrics = ['map', 'ndcg']

true_dict = {'Homer': {'salad': 10,
                  'God': 9,
                  'fat': 8,
                  'Married': 7,
                  'far': 6,
                  'moron': 5,
                  'wife': 4,
                  'medical': 3,
                  'extra': 2,
                  'dammit': 1},
            'Marge': {
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

df_clean = Pre.load_corpus(path_to_file)
sentences = genf.phraser_train(df_clean)
model_list = genf.build_model(w, d, cores, sentences, output_path)
pytrec_dict, filename = EW.calculate_sim(output_path, w, d, true_dict, metrics, "homer", sim_name='cos')
vs.visualize(pytrec_dict, evaluation_charts, filename)
pytrec_dict = EW.calculate_sim(output_path, w, d, true_dict, metrics, "homer", sim_name='man')
vs.visualize(pytrec_dict, evaluation_charts, filename)
pytrec_dict = EW.calculate_sim(output_path, w, d, true_dict, metrics, "homer", sim_name='euc')
vs.visualize(pytrec_dict, evaluation_charts, filename)

brown_sents = brown.sents(categories='news')
model_list_brown = genf.build_model(w, d, cores, brown_sents, brown_path)

word_dic = GS.process(f'{brown_path}{simFile}')

# Up to Here
# Print Number of missing words from our Corpus when compared with GoldStandard
GS.missingWords(word_dic, f'{output_path}w2v_model_window-2_size-10.mdl')


pytrec_dict = EW.evaluate(output_path, w, d, distance, metrics, './evaluation/results/', word_dic)
vs.visualize(pytrec_dict, evaluation_charts, 'Comparison Against WordSim GoldStandard')
