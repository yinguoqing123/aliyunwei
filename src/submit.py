import numpy as np
import pandas as pd
import json 
import os

n = 5
df = pd.read_csv(f"../submit{n}/submit_0.csv")

# for i in range(1, 10):
#     tmp = pd.read_csv(f"../submit{n}/submit_{i}.csv")[['sn', f'label_{i}']]
#     df = df.merge(tmp, on='sn', how='inner')

n = 5
i = 9
df = pd.read_csv(f"../submit{n}/submit_{i}_pos_5m.csv")

def score(x):
    label_0, label_1, label_2, label_3, label_4 = np.array(json.loads(x.label_0)), np.array(json.loads(x.label_1)), np.array(json.loads(x.label_2)), np.array(json.loads(x.label_3)), np.array(json.loads(x.label_4))
    label = (label_0 + label_1 + label_2 + label_3 + label_4)/5
    label = label.argmax()
    return label

def score1(x):
    label_0, label_1, label_2, label_3, label_4 = np.array(json.loads(x.label_0)), np.array(json.loads(x.label_1)), np.array(json.loads(x.label_2)), np.array(json.loads(x.label_3)), np.array(json.loads(x.label_4))
    label_5, label_6, label_7, label_8, label_9 = np.array(json.loads(x.label_5)), np.array(json.loads(x.label_6)), np.array(json.loads(x.label_7)), np.array(json.loads(x.label_8)), np.array(json.loads(x.label_9))
    #label = label_0**2 + label_1**2 + label_2**2 + label_3**2 + label_4**2 + \
    #    label_5**2 + label_6**2 + label_7**2 + label_8**2 + label_9**2
    label = (label_0**2 + label_1**2 + label_2**2 + label_3**2 + label_4**2 + label_5**2 + label_6**2 + label_7**2 + label_8**2 + label_9**2) / (1+label_0+label_1+label_2+label_3+label_4 +  label_5 + label_6 + label_7 + label_8 + label_9)
    label = label.argmax()
    return label

def score2(x):
    label_0, label_1, label_2, label_3, label_4 = np.array(json.loads(x.label_0)), np.array(json.loads(x.label_1)), np.array(json.loads(x.label_2)), np.array(json.loads(x.label_3)), np.array(json.loads(x.label_4))
    label_5, label_6, label_7, label_8, label_9 = np.array(json.loads(x.label_5)), np.array(json.loads(x.label_6)), np.array(json.loads(x.label_7)), np.array(json.loads(x.label_8)), np.array(json.loads(x.label_9))
    #label = label_0**2 + label_1**2 + label_2**2 + label_3**2 + label_4**2 + \
    #    label_5**2 + label_6**2 + label_7**2 + label_8**2 + label_9**2
    label = (label_0**2 + label_1**2 + label_2**2 + label_3**2 + label_4**2 + label_5**2 + label_6**2 + label_7**2 + label_8**2 + label_9**2) / (1+label_0+label_1+label_2+label_3+label_4 +  label_5 + label_6 + label_7 + label_8 + label_9)
    return json.dumps(list(label))

def score3(x):
    label1, label2, label3 = np.array(json.loads(x.label)), np.array(json.loads(x.label2)), np.array(json.loads(x.label3))
    label = (label1 + label2 + label3).argmax()
    return label

#df['label'] = df.apply(score, axis=1)
df['label'] = df.apply(score1, axis=1)

#df2 = pd.read_csv(f"../submit{n}/submit_{i}_pos.csv")
#df3 = pd.read_csv(f"../submit{n}/submit_{i}.csv")

#df2['label2'] = df2.apply(score2, axis=1)
#df3['label3'] = df3.apply(score2, axis=1)

# df = df[['sn', 'fault_time', 'label']].merge(df2[['sn', 'fault_time', 'label2']], on=['sn', 'fault_time'], how='inner')\
#     .merge(df3[['sn', 'fault_time', 'label3']], on=['sn', 'fault_time'], how='inner')
    
# df['label'] = df.apply(lambda x: score3(x), axis=1)

# df[['sn', 'fault_time', 'label']].to_csv(f'../submit{n}/submit.csv', index=False)

# df = df[['sn', 'fault_time', 'label1']]
# df.columns = ['sn', 'fault_time', 'label']
df[['sn', 'fault_time', 'label']].to_csv(f"../submit{n}/submit.csv", index=False)