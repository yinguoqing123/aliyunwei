import numpy as np
import pandas as pd
import json 
import os

n = 5
df = pd.read_csv(f"../submission/submit_9.csv")
# submit_df = pd.read_csv("/tcdata/final_submit_dataset_a.csv")[['sn', 'fault_time']]

def score(x):
    label_0, label_1, label_2, label_3, label_4 = np.array(json.loads(x.label_0)), np.array(json.loads(x.label_1)), np.array(json.loads(x.label_2)), np.array(json.loads(x.label_3)), np.array(json.loads(x.label_4))
    label = (label_0 + label_1 + label_2 + label_3 + label_4)/5
    label = label.argmax()
    return label

def score1(x):
    label_0, label_1, label_2, label_3, label_4 = np.array(json.loads(x.label_0)), np.array(json.loads(x.label_1)), np.array(json.loads(x.label_2)), np.array(json.loads(x.label_3)), np.array(json.loads(x.label_4))
    label_5, label_6, label_7, label_8, label_9 = np.array(json.loads(x.label_5)), np.array(json.loads(x.label_6)), np.array(json.loads(x.label_7)), np.array(json.loads(x.label_8)), np.array(json.loads(x.label_9))
    label = (label_0**2 + label_1**2 + label_2**2 + label_3**2 + label_4**2 + label_5**2 + label_6**2 + label_7**2 + label_8**2 + label_9**2) / (1+label_0+label_1+label_2+label_3+label_4 +  label_5 + label_6 + label_7 + label_8 + label_9)
    label = label.argmax()
    return label


df['label'] = df.apply(score1, axis=1)

# print("submit df shape", submit_df.shape)
# print("df shape", df.shape)
# submit_df = submit_df[~submit_df.sn.isin(df.sn)]
# submit_df['label'] = 2
# df = pd.concat([df[['sn', 'fault_time', 'label']], submit_df[['sn', 'fault_time', 'label']]])
df[['sn', 'fault_time', 'label']].to_csv(f"../submission/submit.csv", index=False)