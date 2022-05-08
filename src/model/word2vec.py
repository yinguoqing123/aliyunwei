import logging
import os.path
from re import S
import sys
import multiprocessing
from gensim.models import Word2Vec
import pandas as pd
import json

# train_log = pd.read_csv(dir + "preliminary_sel_log_dataset.csv")
# train_df = pd.read_csv(dir + 'preliminary_train_label_dataset.csv')
# submit_df = pd.read_csv(dir + 'preliminary_submit_dataset_b.csv')
# train_log_a = pd.read_csv(dir + "preliminary_sel_log_dataset_b.csv")
# train_df_a = pd.read_csv(dir + 'preliminary_train_label_dataset_s.csv')
# train_log = pd.concat([train_log, train_log_a])
# train_df = pd.concat([train_df, train_df_a])
# train_df = train_df.drop_duplicates()

df = pd.read_csv("../data/train_set.csv")

def get_sentence(feat_seq):
    sentence = []
    for feat in json.loads(feat_seq):
        sentence.append('_'.join([str(i) for i in feat[:3]]))
    return sentence

sentences = list(df.feature.apply(get_sentence))
model = Word2Vec(sentences, sg=1, vector_size=36, window=5, min_count=1, negative=1, 
                 sample=0.001, workers=4, epochs=3)
model.save('../data/word2vec.bin')




