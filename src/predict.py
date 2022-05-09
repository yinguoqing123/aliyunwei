# 序列格式微调  加入位置position  
import torch
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
import numpy as np
from torchtext.vocab import vocab
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import json
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from torch.optim.lr_scheduler import *
from model.utils import macro_f1, FGM
from model.dataset import MyDataSet
from model.model import MyModel
import random
import os
import warnings
warnings.filterwarnings("ignore")

test_set = pd.read_csv('../tmp_data/test_set_b.csv')

test_set_ = MyDataSet(test_set, mode='predict')
test_set_iter = iter(test_set_)
model = MyModel()


for fold in range(10):
    model.load_state_dict(torch.load(f'../model/model_{fold}.pt'))
    preds = []
    with torch.no_grad():
        model.eval()
        for step in range(test_set_.step_max):
            feat = next(test_set_iter)
            pred = model(feat)
            pred = torch.softmax(pred, dim=-1).numpy()
            pred = [json.dumps(p.tolist()) for p  in pred]
            preds.extend(pred)
    test_set[f'label_{fold}'] = preds
    
def score(x):
    label_0, label_1, label_2, label_3, label_4 = np.array(json.loads(x.label_0)), np.array(json.loads(x.label_1)), np.array(json.loads(x.label_2)), np.array(json.loads(x.label_3)), np.array(json.loads(x.label_4))
    label_5, label_6, label_7, label_8, label_9 = np.array(json.loads(x.label_5)), np.array(json.loads(x.label_6)), np.array(json.loads(x.label_7)), np.array(json.loads(x.label_8)), np.array(json.loads(x.label_9))
    label = (label_0 + label_1 + label_2 + label_3 + label_4 + label_5 + label_6 +label_7 + label_8 + label_9)/10
    # label = label.argmax()
    label = label.max()
    return label

def score1(x):
    label_0, label_1, label_2, label_3, label_4 = np.array(json.loads(x.label_0)), np.array(json.loads(x.label_1)), np.array(json.loads(x.label_2)), np.array(json.loads(x.label_3)), np.array(json.loads(x.label_4))
    label_5, label_6, label_7, label_8, label_9 = np.array(json.loads(x.label_5)), np.array(json.loads(x.label_6)), np.array(json.loads(x.label_7)), np.array(json.loads(x.label_8)), np.array(json.loads(x.label_9))
    label = (label_0**2 + label_1**2 + label_2**2 + label_3**2 + label_4**2 + label_5**2 + label_6**2 + label_7**2 + label_8**2 + label_9**2) / (1+label_0+label_1+label_2+label_3+label_4 +  label_5 + label_6 + label_7 + label_8 + label_9)
    label = label.argmax()
    return label


test_set['label'] = test_set.apply(score1, axis=1)
test_set['positive_p'] = test_set.apply(score, axis=1)

# print("submit df shape", submit_df.shape)
# print("df shape", df.shape)
# submit_df = submit_df[~submit_df.sn.isin(df.sn)]
# submit_df['label'] = 2
# df = pd.concat([df[['sn', 'fault_time', 'label']], submit_df[['sn', 'fault_time', 'label']]])
test_set[['sn', 'fault_time', 'label', 'positive_p']].to_csv(f"../submission/submit.csv", index=False)
