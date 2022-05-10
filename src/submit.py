import numpy as np
import pandas as pd
import json 
import os


df = pd.read_csv(f"../submission/submit.csv")
submit_df = pd.read_csv("/tcdata/final_submit_dataset_a.csv")[['sn', 'fault_time']]
print("df shape:", df.shape)
print("submit_df shape:", submit_df.shape)
print("submit中存在df中不存在:", submit_df[~submit_df.sn.isin(df.sn)].shape)

def score(x):
    score = np.array([np.array(json.loads(score)) for score in x])
    score = (score**2).sum(axis=0) / (1 + score.sum(axis=0))
    label = score.argmax()
    return label

col = [col for col in df.columns if 'label_' in col]
label = df[col].apply(lambda x: score(x), axis=1)
df['label'] = label

df[['sn', 'fault_time', 'label']].to_csv(f"../submission/submit.csv", index=False)
