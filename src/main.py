import imp
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
from utils import macro_f1, MyDataSet

dir = r'D:\ai-risk\aliyunwei\data'

train_df = pd.read_csv("../data/train_set.csv")
submit_df = pd.read_csv('../data/submit_df.csv')

lookup1 = torch.load('word2idx.pk')

servertype2id = dict()
servertypes = sorted(list(set(train_df.server_model)))
for servertype in servertypes:
    servertype2id[servertype] = len(servertype2id) + 1
servertype2id['pad'] = 0

                
class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.emb1 = nn.Embedding(len(lookup1), 32, padding_idx=0)
        self.emb2 = nn.Embedding(len(servertype2id), 32, padding_idx=0)
        self.lstm1 = nn.LSTM(32, 32, batch_first=True)
        self.lstm2 = nn.LSTM(32, 32, batch_first=True)
        self.bn = nn.LayerNorm(32)
        self.classify = nn.Linear(64, 4)
        #self.classify.bias.data = torch.tensor([-2.38883658, -1.57741002, -0.57731536, -1.96360971])
        
    def forward(self, feat):
        feat, server_model, len1, len2 = feat  # len1 batch_size * sentence_num  ||   len2 batch_size
        word_emb = self.emb1(feat)  # batch_size * sentence_num * word_num * emb_dim
        server_model = self.emb2(server_model)
        word_emb = word_emb.reshape(-1, word_emb.shape[-2], word_emb.shape[-1])  # (batch_size * sentence_num , word_num(step), emb_dim)
        word_emb, _ = self.lstm1(word_emb)
        len1 = len1.reshape(-1)
        interval_range = torch.arange(0, feat.shape[0]*feat.shape[1]*feat.shape[2], feat.shape[2]) + len1 - 1
        sentence_emb = torch.index_select(word_emb.reshape(-1,  word_emb.shape[-1]), 0, interval_range).reshape(feat.shape[0], feat.shape[1], -1)  # batch_size * sentence_num * emb_dim
        sentence_emb = self.bn(sentence_emb)
        sentence_emb, _ = self.lstm2(sentence_emb) # batch_size * sentence_num * emb_dim
        interval_range2 = torch.arange(0, feat.shape[0]*feat.shape[1], feat.shape[1]) + len2 - 1 
        final_emb = torch.index_select(sentence_emb.reshape(-1, word_emb.shape[-1]), 0, interval_range2).reshape(feat.shape[0], -1)  # batch_size * emb_dim
        score = self.classify(torch.concat([final_emb, server_model], dim=-1))
        score = F.softmax(F.relu(score), dim=-1)  # batch_size * 4
        return score

train_df = train_df.sample(frac=1, random_state=2022).reset_index(drop=True)         


model = Model()
def lr_lambda(epoch):
    if epoch > 15:
        return 0.1
    else:
        return 1

optimizer = optim.Adam(model.parameters(), lr=1e-3)
lr_sche = LambdaLR(optimizer, lr_lambda)
def train_and_evaluate(train_set_, test_set_, submit_set_, name):
    model = Model() 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lr_sche = LambdaLR(optimizer, lr_lambda)
    train_set_ = MyDataSet(train_set_)
    test_df = test_set_
    test_set_ = MyDataSet(test_set_)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 2.0, 1.0, 1.0]))
    train_data = iter(train_set_)
    test_data = iter(test_set_)
    best_f1 = 0
    for epoch in range(20):
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
            torch.save(model.state_dict(), f'../model/model_{name}.pt')
        test_df.to_csv(f"../valid/pred_{name}.csv", index=False)
        print(f"macro F1: {macro_F1}")
    submit_set = MyDataSet(submit_set_, mode='predict')
    submit_set_iter = iter(submit_set)
    preds = []
    model.load_state_dict(torch.load(f'../model/model_{name}.pt'))
    with torch.no_grad():
        model.eval()
        for step in range(submit_set.step_max):
            feat = next(submit_set_iter)
            pred = model(feat).numpy()
            pred = [json.dumps(p.tolist()) for p  in pred]
            preds.extend(pred)
    submit_set_[f'label_{name}'] = preds
    submit_set_.to_csv(f'../submit/submit_{name}.csv', index=False)
    

for i, (train_idx, test_idx) in enumerate(KFold().split(train_df[['sn', 'fault_time','feature', 'servertype', 'label']])):
    train_set = train_df.iloc[train_idx]
    train_set = pd.concat([train_set, train_set[train_set.label==0]]).reset_index(drop=True)
    test_set = train_df.iloc[test_idx]     
    train_and_evaluate(train_set, test_set, submit_df, i)



            
 