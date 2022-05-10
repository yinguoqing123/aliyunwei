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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(2022)

train_set = pd.read_csv("../tmp_data/train_set.csv")
finala_set = pd.read_csv("../tmp_data/finala_set.csv")
# finala_set = pd.read_csv('../tmp_data/test_set_a.csv')


def train_and_evaluate(train_set_, test_set_, submit_set_, name, att_cate='pool'):
    model = MyModel(att_cate=att_cate) 
    fgm = FGM(model)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2)
    #lr_sche = LambdaLR(optimizer, lr_lambda)
    train_set_ = MyDataSet(train_set_)
    test_df = test_set_
    test_set_ = MyDataSet(test_set_, mode='test')
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 2.0, 1.0, 1.0]))
    train_data = iter(train_set_)
    test_data = iter(test_set_)
    best_f1 = 0
    for epoch in range(40):
        running_loss = 0
        for step in range(train_set_.step_max):
            feat, label = next(train_data)
            model.train()
            pred = model(feat)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            fgm.attack() 
            loss_adv = criterion(model(feat), label)
            loss_adv.backward() 
            fgm.restore() 
            optimizer.step()
            running_loss += loss.item()
            if step % 50 == 49:
                print(f"Epoch {epoch+1}, step {step+1}: {running_loss}")
                running_loss = 0 
        #lr_sche.step()   
        # epoch 结束后 evaluate
        preds = []
        with torch.no_grad():
            for step in range(test_set_.step_max):
                model.eval()
                feat, label = next(test_data)
                pred = model(feat).argmax(dim=-1).numpy()
                preds.extend(pred)
        test_df['pred'] = preds
        macro_F1 =  macro_f1(test_df)
        if macro_F1 > best_f1:
            torch.save(model.state_dict(), f'../model/model_{att_cate}_{name}.pt')
            best_f1 = macro_F1
            test_df.to_csv(f"../submission/pred.csv", index=False)
        print(f"macro F1: {macro_F1}")
        print(f"max macro F1: {best_f1}")
        scheduler.step(macro_F1)
    print('max macro F1:', best_f1)
    submit_set = MyDataSet(submit_set_, mode='predict')
    submit_set_iter = iter(submit_set)
    preds = []
    model.load_state_dict(torch.load(f'../model/model_{att_cate}_{name}.pt'))
    with torch.no_grad():
        model.eval()
        for step in range(submit_set.step_max):
            feat = next(submit_set_iter)
            pred = model(feat)
            pred = torch.softmax(pred, dim=-1).numpy()
            pred = [json.dumps(p.tolist()) for p  in pred]
            preds.extend(pred)
    submit_set_[f'label_{att_cate}_{name}'] = preds
    submit_set_.to_csv(f'../submission/submit.csv', index=False)
    
for i, (train_idx, test_idx) in enumerate(StratifiedKFold(n_splits=10, shuffle=True, random_state=2021).split(train_set, train_set.label)):
    train_set_ = train_set.iloc[train_idx]
    train_set_ = pd.concat([train_set_, train_set_[train_set_.label==0]]).reset_index(drop=True)
    test_set_ = train_set.iloc[test_idx]     
    train_and_evaluate(train_set_, test_set_, finala_set, i, att_cate='gate')
    print('=====================================')
    
print("=========================  模型1训练结束  ===========================")

for i, (train_idx, test_idx) in enumerate(StratifiedKFold(n_splits=10, shuffle=True, random_state=2022).split(train_set, train_set.label)):
    train_set_ = train_set.iloc[train_idx]
    train_set_ = pd.concat([train_set_, train_set_[train_set_.label==0]]).reset_index(drop=True)
    test_set_ = train_set.iloc[test_idx]     
    train_and_evaluate(train_set_, test_set_, finala_set, i)
    print('=====================================')
    
print("=========================  模型2训练结束  ===========================")