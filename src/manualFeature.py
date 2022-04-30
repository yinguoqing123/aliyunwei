# 构造人工特征
# 人工特征统计
import pandas as pd
import numpy as np
from collections import Counter
import json
import torch 

train_set = pd.read_csv("../data/train_set4.csv")
df = train_set.copy(deep=True)

df['feature1'] = df.feature.apply(lambda x: np.array(json.loads(x)).flatten().tolist())

def calfeat1(i):
    class_i = []
    for msgs in list(df[df.label==i].feature1):
        class_i.extend(msgs) 
    
    class_i = Counter(class_i)
    v = sum(class_i.values())
    for key in class_i:
        class_i[key] = class_i[key]/v

class_0 = calfeat1(0)
class_1 = calfeat1(1)
class_2 = calfeat1(2)
class_3 = calfeat1(3)

lookup1 = torch.load('../data/word2idx3.pk')

featmatrix1 = []
for i in range(len(lookup1)):
    feat = []
    feat.append(class_0.get(i, 0.0))
    feat.append(class_1.get(i, 0.0))
    feat.append(class_2.get(i, 0.0))
    feat.append(class_3.get(i, 0.0))
    featmatrix1.append(feat)
    
# ========================

df['feature2'] = df.feature.apply(lambda x: set(np.array(json.loads(x)).flatten().tolist()))

wordstatics = {}

for feat, label in zip(list(df.feature2), list(df.label)):
    for word in feat:
        if word not in wordstatics:
            wordstatics[word] = [0.0, 0.0, 0.0, 0.0]
        else:
            wordstatics[word][label] += 1
          
prior_mean = np.array([0.089383, 0.204945, 0.560175, 0.145498])
for key in wordstatics:
    weights = np.array(wordstatics[key]) / (np.array(wordstatics) + 40)
    wordstatics[key] = weights * np.array(wordstatics[key])/(sum(wordstatics[key])+1e-12) +  (1 - weights) * prior_mean

featmatrix2 = []
for i in range(len(lookup1)):
    if i not in wordstatics:
        featmatrix2.append([0.0, 0.0, 0.0, 0.0])
    else:
        featmatrix2.append(wordstatics[i])
        
np.save('../data/featmatrix1.npy', np.array(featmatrix1))
np.save('../data/featmatrix2.npy', np.array(featmatrix2))
        
