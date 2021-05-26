# Overview

This project focuses on comparing the results of Word2Vec using Gensim implementation and finding out the ideal window size and dimensionality for obtaining best similarity results. The similarity results were calculated using Cosine, Eculidean, and Manhattan metrics. The results were then compared with the Goldstandard True Words. Information Retrieval metrics were used such as MAP and NDCG scores to measure the similartity between true words and words from our Simpsons Dataset.

# Pre-Requisite

1. Gensim Library: - !pip install gensim
2. Pytrec Eval: - !pip install pytrec_eval
3. NLTK Brown Corpus - from nltk.corpus import brown
4. Spacy - !pip install spacy
5. Simpsons Dataset from Kaggle
6. WordSim GoldStandard Dataset - Download it from http://alfonseca.org/pubs/ws353simrel.tar.gz." and Unzip wordsim_similarity_goldstandard into the evaluation folder.

# Running the Code
python main.py

# Output

This projects works on finding the optimal hyperparameters set for Word2Vec and then compares it with WordSim GoldStandard True Words. Graphs of simialrit metrics cosine, euclidean and manhattan and iformation retrieval metrics such as map and ndcg is created after a successful execution.
