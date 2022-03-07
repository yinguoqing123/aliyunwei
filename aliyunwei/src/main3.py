
import imp
from unicodedata import bidirectional
from numpy import pad
import torch
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
import numpy as np
from torchtext.vocab import vocab
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import json
from sklearn.model_selection import train_test_split, KFold
from ema import EMA
from torch.optim.lr_scheduler import *
from utils import macro_f1, MyDataSet3

train_set = pd.read_csv('../data/train_set3.csv')
submit_df = pd.read_csv('../data/submit_df3.csv')

# train_set, test_set = train_test_split(train_set, test_size=0.2, random_state=2022)
# cols = ['sn', 'fault_time', 'servertype', 'feature', 'label']
# train_data = MyDataSet3(train_set[cols])
# test_data = MyDataSet3(test_set[cols])
lookup1 = torch.load('../data/word2idx3.pk')


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.emb1 = nn.Embedding(len(lookup1), 16, padding_idx=0)
        self.emb2 = nn.Embedding(88, 16, padding_idx=0)
        self.lstm = nn.LSTM(48, 32, batch_first=True, bidirectional=False)
        self.classify = nn.Linear(48, 4)
        self.orthogonal_weights()
        #self.classify.bias.data = torch.tensor([-2.38883658, -1.57741002, -0.57731536, -1.96360971])
        
    def forward(self, feat):
        feat, server_model, len_seq = feat  # len1 batch_size * sentence_num
        word_emb = self.emb1(feat)  # batch_size * sentence_num * word_num * emb_dim
        b, s, w, d = word_emb.shape
        server_model = self.emb2(server_model)
        #server_model_clone = torch.clone(server_model).reshape(b, 1, -1).tile(s, 1)
        word_emb = word_emb.reshape(b, s, w*d)
        #word_emb = torch.cat([word_emb, server_model_clone], dim=-1)
        word_emb_pack = pack_padded_sequence(word_emb, len_seq, batch_first=True,enforce_sorted=False)
        word_emb, _ = self.lstm(word_emb_pack) 
        word_emb, _ = pad_packed_sequence(word_emb, batch_first=True) # b, s, d
        interval_range = torch.arange(0, b*s, s) + len_seq - 1 
        word_emb = torch.index_select(word_emb.reshape(b*s, -1), 0, interval_range).reshape(b, -1)  # batch_size * emb_dim
        score = self.classify(torch.concat([word_emb, server_model], dim=-1))
        score = F.softmax(score, dim=-1)  # batch_size * 4
        return score

model = Model()
def lr_lambda(epoch):
    if epoch > 20:
        return 0.1
    else:
        return 1
    
model = Model()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
lr_sche = LambdaLR(optimizer, lr_lambda)
def train_and_evaluate(train_set_, test_set_, submit_set_, name):
    model = Model() 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lr_sche = LambdaLR(optimizer, lr_lambda)
    train_set_ = MyDataSet3(train_set_)
    test_df = test_set_
    test_set_ = MyDataSet3(test_set_, mode='test')
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 2.0, 1.0, 1.0]))
    train_data = iter(train_set_)
    test_data = iter(test_set_)
    best_f1 = 0
    for epoch in range(30):
        running_loss = 0
        for step in range(train_set_.step_max):
            feat, label = next(train_data)
            model.train()
            pred = model(feat)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % 50 == 49:
                print(f"Epoch {epoch+1}, step {step+1}: {running_loss}")
                running_loss = 0 
        lr_sche.step()   
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
            torch.save(model.state_dict(), f'../model3/model_{name}.pt')
        test_df.to_csv(f"../valid3/pred_{name}.csv", index=False)
        print(f"macro F1: {macro_F1}")
    submit_set = MyDataSet3(submit_set_, mode='predict')
    submit_set_iter = iter(submit_set)
    preds = []
    model.load_state_dict(torch.load(f'../model3/model_{name}.pt'))
    with torch.no_grad():
        model.eval()
        for step in range(submit_set.step_max):
            feat = next(submit_set_iter)
            pred = model(feat).numpy()
            pred = [json.dumps(p.tolist()) for p  in pred]
            preds.extend(pred)
    submit_set_[f'label_{name}'] = preds
    submit_set_.to_csv(f'../submit3/submit_{name}.csv', index=False)
    

for i, (train_idx, test_idx) in enumerate(KFold().split(train_set[['sn', 'fault_time','feature', 'servertype', 'label']])):
    train_set_ = train_set.iloc[train_idx]
    train_set_ = pd.concat([train_set_, train_set_[train_set_.label==0]]).reset_index(drop=True)
    test_set_ = train_set.iloc[test_idx]     
    train_and_evaluate(train_set_, test_set_, submit_df, i)