# Overview

This project focuses on comparing the results of Word2Vec using Gensim implementation and finding out the ideal window size and dimensionality for obtaining best similarity results. The similarity results were calculated using Cosine, Eculidean, and Manhattan metrics. The results were then compared with the Goldstandard True Words. Information Retrieval metrics were used such as MAP and NDCG scores to measure the similartity between true words and words from our Simpsons Dataset.

# Pre-Requisite

1. Gensim Library: - !pip install gensim
2. Pytrec Eval: - !pip install pytrec_eval
3. NLTK Brown Corpus - from nltk.corpus import brown
4. Spacy - !pip install spacy
5. Simpsons Dataset from Kaggle
