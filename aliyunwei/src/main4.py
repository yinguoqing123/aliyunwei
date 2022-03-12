# 序列格式微调
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
from utils import MyDataSet4, macro_f1, DiceLoss

train_set = pd.read_csv('../data/train_set4.csv')
submit_df = pd.read_csv('../data/submit_df4.csv')

lookup1 = torch.load('../data/word2idx4.pk')

intervalbucketnum, cntbucketnum, durationbucketnum = 101, 23, 12

class AttentionPooling1D(nn.Module):
    """通过加性Attention,将向量序列融合为一个定长向量
    """
    def __init__(self, in_features,  **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.in_features = in_features # 词向量维度
        self.k_dense = nn.Linear(self.in_features, self.in_features, bias=False)
        self.o_dense = nn.Linear(self.in_features, 1, bias=False)
    def forward(self, inputs):
        xo, mask = inputs
        mask = mask.unsqueeze(-1)
        x = self.k_dense(xo)
        x = self.o_dense(F.tanh(x))  # N, s, 1
        x = x - (1 - mask) * 1e12
        x = F.softmax(x, dim=-2)  # N, w, 1
        return torch.sum(x * xo, dim=-2) # N*emd

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.emb1 = nn.Embedding(len(lookup1), 16, padding_idx=0)
        self.emb2 = nn.Embedding(88, 16, padding_idx=0)
        self.emb3 = nn.Embedding(intervalbucketnum, 8)
        self.emb4 = nn.Embedding(cntbucketnum, 4)
        self.emb5 = nn.Embedding(durationbucketnum, 4)
        self.lstm = nn.LSTM(64, 32, batch_first=True, bidirectional=False)
        self.att = AttentionPooling1D(64)
        self.classify = nn.Linear(116, 4)
        self.orthogonal_weights()
        self.classify.bias.data = torch.tensor([-2.38883658, -1.57741002, -0.57731536, -1.96360971])

        
    def forward(self, feat):
        feat, server_model, len_seq, mask = feat  # len1 batch_size * sentence_num
        feat1, feat2, feat3, feat4, feat5 = feat[..., :3].int(), feat[..., -3].int(), feat[..., -2].int(), feat[..., -1].int(), feat[..., 3:7]  # (b, s, 3), (b, s) (b, s) (b, s)
        word_emb = self.emb1(feat1)  # (b, s, 3, d)
        emd_interval = self.emb3(feat2)
        emb_cnt = self.emb4(feat3)
        emb_duration = self.emb5(feat4)
        b, s, w, d = word_emb.shape
        server_model = self.emb2(server_model)
        word_emb = word_emb.reshape(b, s, w*d)
        word_emb = torch.concat([word_emb, emd_interval, emb_cnt, emb_duration], dim=-1)
        # word_emb attention
        att_emb = self.att((word_emb, mask))
        word_emb_pack = pack_padded_sequence(word_emb, len_seq, batch_first=True,enforce_sorted=False)
        word_emb, _ = self.lstm(word_emb_pack) 
        word_emb, _ = pad_packed_sequence(word_emb, batch_first=True) # b, s, d
        interval_range = torch.arange(0, b*s, s) + len_seq - 1 
        word_emb = torch.index_select(word_emb.reshape(b*s, -1), 0, interval_range).reshape(b, -1)  # batch_size * emb_dim
        feat5 =  torch.sum(feat5, dim=-2)/torch.tile(len_seq.unsqueeze(dim=-1), (4,)) # (b, s, 4) --> (b, 4)
        score = self.classify(torch.concat([word_emb, server_model, att_emb, feat5], dim=-1))
        
        return score
    
    def orthogonal_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                torch.nn.init.orthogonal_(param)
            if 'bias' in name:
                torch.nn.init.zeros_(param)
                
    def manual_embeddings(self, featmatrix1, featmatrix2):
        self.emb3 = nn.Embedding(len(lookup1), 4, padding_idx=0)
        self.emb4 = nn.Embedding(len(lookup1), 4, padding_idx=0)
        self.emb3.weight.requires_grad = False
        self.emb3.weight.data = featmatrix1
        self.emb4.weight.requires_grad = False
        self.emb4.weight.data = featmatrix2


model = Model()
def train_and_evaluate(train_set_, test_set_, submit_set_, name):
    model = Model() 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)
    #lr_sche = LambdaLR(optimizer, lr_lambda)
    train_set_ = MyDataSet4(train_set_)
    test_df = test_set_
    test_set_ = MyDataSet4(test_set_, mode='test')
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 2.0, 1.0, 1.0]))
    #criterion = DiceLoss(weight=torch.tensor([1.5, 1.5, 1.0, 1.0]), coff1=1, coff2=0.5)
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
            torch.save(model.state_dict(), f'../model4/model_{name}.pt')
        test_df.to_csv(f"../valid4/pred_{name}.csv", index=False)
        print(f"macro F1: {macro_F1}")
        scheduler.step(macro_F1)
    submit_set = MyDataSet4(submit_set_, mode='predict')
    submit_set_iter = iter(submit_set)
    preds = []
    model.load_state_dict(torch.load(f'../model4/model_{name}.pt'))
    with torch.no_grad():
        model.eval()
        for step in range(submit_set.step_max):
            feat = next(submit_set_iter)
            pred = model(feat)
            pred = torch.softmax(pred, dim=-1).numpy()
            pred = [json.dumps(p.tolist()) for p  in pred]
            preds.extend(pred)
    submit_set_[f'label_{name}'] = preds
    submit_set_.to_csv(f'../submit4/submit_{name}.csv', index=False)
    
for i, (train_idx, test_idx) in enumerate(KFold(shuffle=True, random_state=2022).split(train_set[['sn', 'fault_time','feature', 'servertype', 'label']])):
    train_set_ = train_set.iloc[train_idx]
    train_set_ = pd.concat([train_set_, train_set_[train_set_.label==0]]).reset_index(drop=True)
    test_set_ = train_set.iloc[test_idx]     
    train_and_evaluate(train_set_, test_set_, submit_df, i)
    print('=====================================')