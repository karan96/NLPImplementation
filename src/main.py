#0. please note that I'm not your mentor. I am your supervisor. There is a huge difference!
#0. please choose a better name for this project. something like w2v_sim_search. The current name has nothing to do with this project.

import pandas as pd
import PreProcessing as Prep
import GensimFunc as GenF #see my comments in this file
import ExtractWords as EW #see my comments in this file
import GoldStandardProc as GS
import SecondCorpus as SC

#1. there is a bug in this line due to use of "\'"
#2. please use relative path to the project. never use absolute path to your local system. other user may not have same path in their systems
#2.1. use forward slash (/) please so that if works in all platforms such windows/linux/mac
path_to_file = 'D:\Desktop\Papers\w2v_project\'


# #3. there should be a folder /evaluation where you put anything related to evaluation of your models such as the gold standards.
# simFile = "./evaluation/wordsim_similarity_goldstandard.txt"
#
# wordsim_df = pd.read_csv(simFile,sep='\t',header=None)
#
# #4. what is "Animal_1" and "Animal_2"? I think you borrowed this from our sample "Tiger". but this is not true in general case. put a better name please like pair_a, pair_b
# wordsim_df.columns = ['Animal_1','Animal_2','Sim_Score']
#

df_clean = Prep.first()

sentences = GenF.Phraser(df_clean)

#w and d are the settings of your pipeline. it should define outside the functions.
#also I can see that you define it multiple times in different files like in GensimFunc.py and ExtractWords.py which is not good.
#I changed it for you as follows. I pass them as argument
cores = multiprocessing.cpu_count()
w = list(range(2, 11))
d = list(range(10, 100, 10)) + list(range(100, 600, 100))

model_list = GenF.build_model(sentences, w, d, cores, './output/w2v_models/')

# I did some changes to your EW file

EW.calculate_sim('./output/w2v_similarity/', w, d, "homer", sim_name='cos')
EW.calculate_sim('./output/w2v_similarity/', w, d, "homer", sim_name='man')
EW.calculate_sim('./output/w2v_similarity/', w, d, "homer", sim_name='euc')

# EW.cosine('./output/w2v_similarity/', w, d, "homer")
# EW.euc('./output/w2v_similarity/', w, d, "homer")
# EW.manh('./output/w2v_similarity/', w, d, "homer")

# we don't need these functions.
# EW.map_cos()
#
# EW.map_euc()
#
# EW.map_man()
#
# EW.ndcg_cos()
#
# EW.ndcg_euc()
#
# EW.ndcg_manh()

word_dic = GS.preprocess("./evaluation/wordsim_similarity_goldstandard.txt")

num_missing_words = GS.missingWords(word_dic, f'w2v_model_window-2_size-10.mdl')

avg_dict = GS.goldstand_map(path_to_file, w, d, word_dic)
# plt.figure(figsize=(30, 20))
# names = list(avg_dict.keys())
# values = list(avg_dict.values())
# plt.plot(names, values, color='b')
# plt.xticks(rotation=90, fontsize='small')
# plt.title("Cosine Similarity Metric for Golden Standerd vs Trained Models: MAP")
# plt.show()


#Karan, you don't need a seperate code or function. What if I add a third corpus, you want to add extra code?
# you have to reuse the same functions that your wrote for the first corpus. everything is the same. The only difference is the input/preprocessing of the input.
# you should not copy and paste the same logic and create new functions!
SC.modeltrain(path_to_file)

SC.compare_map(word_dic, path_to_file)

