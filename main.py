import pandas as pd
import PreProcessing as Pre
import GensimFunc as GenF
import ExtractWords as EW

path_to_file = 'D:\Desktop\Papers\wordsim353_sim_rel\'

simFile = path_to_file + "wordsim_similarity_goldstandard.txt"

wordsim_df = pd.read_csv(simFile,sep='\t',header=None)

wordsim_df.columns = ['Animal_1','Animal_2','Sim_Score']


df_clean = Pre.first()

sentences = GenF.Phraser(df_clean)

model_list = GenF.build_model(path_to_file, sentences)

EW.cosine(path_to_file)

EW.euc(path_to_file)

EW.manh(path_to_file)

EW.map_cos()

EW.map_euc()

EW.map_man()

EW.ndcg_cos()

EW.ndcg_euc()

EW.ndcg_manh()