import pandas as pd
import PreProcessing as Pre
import w2v_train as GenF
import w2v_similarity as EW
import GoldStandardProc as GS
import SecondCorpus as SC
import os
import nltk
from nltk.corpus import brown

nltk.download('brown')

# setting directory path
# This File assumes that the files related to project are stored in Evaluation Folder.

data_file = "simpsons_dataset.csv"

simFile = "wordsim_similarity_goldstandard.txt"

# setting up window size range and dimensionality range

cores = multiprocessing.cpu_count()
w = list(range(2, 11))
d = list(range(10, 100, 10)) + list(range(100, 600, 100))

work_dir = os.path.abspath(__file__)

dir_name = os.path.dirname(__file__)

os.chdir(dir_name)

cwd = os.getcwd()


def find(simFile, eval_path):
    for root, dirs, files in os.walk(eval_path):
        if simFile in files:
            print("File Found")
        else:
            print("WordSim Similarity Goldstandard file not found.")
            print(
                "Please download it from http://alfonseca.org/pubs/ws353simrel.tar.gz.")
            print("Unzip wordsim_similarity_goldstandard into the evaluation folder ")
            print("Exiting Program")
            exit()


if os.path.exists('evaluation'):
    print("Path Exists Looking for Wordsim Similarity File")
    find(simFile, eval_path)
else:
    os.mkdir(dir_name + '/evaluation')

eval_path = cwd + '\evaluation'

os.chdir(eval_path)

find(simFile, eval_path)

path_to_file = os.path.join(eval_path, data_file)

# Pre Processing and Training

df_clean = Pre.load_corpus(path_to_file)

sentences = GenF.Phraser(df_clean)

model_list = GenF.build_model(path_to_file, w, d, cores, sentences, eval_path)

EW.calculate_sim(eval_path, w, d, "homer", sim_name='cos')
EW.calculate_sim(eval_path, w, d, "homer", sim_name='man')
EW.calculate_sim(eval_path, w, d, "homer", sim_name='euc')

'''We do not need to calculate map and ndcg values for each metric? But we did calculate it while we were working on 
the project. I will be uploading another piece of reusable code to remove the usage of same code three times for 
calculating map and ndcg scores. '''

EW.map_cos()

EW.map_euc()

EW.map_man()

EW.ndcg_cos()

EW.ndcg_euc()

EW.ndcg_manh()

word_dic = GS.preprocess(eval_path)

GS.missingWords(word_dic, f'w2v_model_window-2_size-10.mdl')

avg_dict = GS.goldstand_map(eval_path)
plt.figure(figsize=(30, 20))
names = list(avg_dict.keys())
values = list(avg_dict.values())
plt.plot(names, values, color='b')
plt.xticks(rotation=90, fontsize='small')
plt.title("Cosine Similarity Metric for Golden Standard vs Trained Models: MAP")
plt.show()

model_list = GenF.build_model(path_to_file, w, d, cores, sentences, eval_path, new_corpus=1)

SC.compare_map(model_list, eval_path)
