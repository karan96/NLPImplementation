import pandas as pd
import PreProcessing as Pre
import GensimFunc as GenF
import ExtractWords as EW
import GoldStandardProc as GS
import SecondCorpus as SC

path_to_file = 'D:\Desktop\Papers\w2v_project\'

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

word_dic = GS.preprocess(path_to_file)

GS.missingWords(path_to_file)

GS.goldstand_map(path_to_file)

SC.modeltrain(path_to_file)

SC.compare_map(word_dic, path_to_file)

