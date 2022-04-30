
import imp
from unicodedata import bidirectional

from scipy import rand

from sklearn.utils import shuffle
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
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
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
featmatrix1 = np.load('../data/featmatrix1.npy')
featmatrix2 = np.load('../data/featmatrix2.npy')

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
    def __init__(self, featmatrix1=None, featmatrix2=None) -> None:
        super(Model, self).__init__()
        self.emb1 = nn.Embedding(len(lookup1), 16, padding_idx=0)
        self.emb2 = nn.Embedding(88, 10, padding_idx=0)
        self.lstm = nn.LSTM(48, 32, batch_first=True, bidirectional=False)
        self.att = AttentionPooling1D(48)
        self.classify = nn.Linear(42, 4)
        self.orthogonal_weights()
        #self.classify.bias.data = torch.tensor([-2.38883658, -1.57741002, -0.57731536, -1.96360971])
        if featmatrix1 is not None and  featmatrix2 is not None:
            self.manual_embeddings(torch.tensor(featmatrix1).float(), torch.tensor(featmatrix2).float())
        
    def forward(self, feat):
        feat, server_model, len_seq, mask = feat  # len1 batch_size * sentence_num
        word_emb = self.emb1(feat)  # batch_size * sentence_num * word_num * emb_dim
        #manual_emb1 = self.emb3(feat)
        #manual_emb2 = self.emb4(feat)
        b, s, w, d = word_emb.shape
        server_model = self.emb2(server_model)
        #server_model_clone = torch.clone(server_model).reshape(b, 1, -1).tile(s, 1)
        word_emb = word_emb.reshape(b, s, w*d)
        #manual_emb1 = manual_emb1.reshape(b, s, w*4)
        #manual_emb2 = manual_emb2.reshape(b, s, w*4)
        #att_emb = self.att((word_emb, mask))
        #word_emb = torch.concat([word_emb, manual_emb1, manual_emb2], dim=-1)
        #word_emb = torch.cat([word_emb, server_model_clone], dim=-1)
        word_emb_pack = pack_padded_sequence(word_emb, len_seq, batch_first=True,enforce_sorted=False)
        word_emb, _ = self.lstm(word_emb_pack) 
        word_emb, _ = pad_packed_sequence(word_emb, batch_first=True) # b, s, d
        interval_range = torch.arange(0, b*s, s) + len_seq - 1 
        word_emb = torch.index_select(word_emb.reshape(b*s, -1), 0, interval_range).reshape(b, -1)  # batch_size * emb_dim
        score = self.classify(torch.concat([word_emb, server_model], dim=-1))
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
def lr_lambda(epoch):
    if epoch > 35:
        return 0.02
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
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=4)
    #lr_sche = LambdaLR(optimizer, lr_lambda)
    train_set_ = MyDataSet3(train_set_)
    test_df = test_set_
    test_set_ = MyDataSet3(test_set_, mode='test')
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
            best_f1 = macro_F1
            torch.save(model.state_dict(), f'../model3/model_{name}.pt')
        test_df.to_csv(f"../valid3/pred_{name}.csv", index=False)
        print(f"macro F1: {macro_F1}")
        scheduler.step(macro_F1)
    print('max macro F1:', best_f1)
    submit_set = MyDataSet3(submit_set_, mode='predict')
    submit_set_iter = iter(submit_set)
    preds = []
    model.load_state_dict(torch.load(f'../model3/model_{name}.pt'))
    with torch.no_grad():
        model.eval()
        for step in range(submit_set.step_max):
            feat = next(submit_set_iter)
            pred = model(feat)
            pred = torch.softmax(pred, dim=-1).numpy()
            pred = [json.dumps(p.tolist()) for p  in pred]
            preds.extend(pred)
    submit_set_[f'label_{name}'] = preds
    submit_set_.to_csv(f'../submit3/submit_{name}.csv', index=False)
    
for i, (train_idx, test_idx) in enumerate(KFold(shuffle=True, random_state=2022).split(train_set[['sn', 'fault_time','feature', 'servertype', 'label']])):
        train_set_ = train_set.iloc[train_idx]
        train_set_ = pd.concat([train_set_, train_set_[train_set_.label==0]]).reset_index(drop=True)
        test_set_ = train_set.iloc[test_idx]     
        train_and_evaluate(train_set_, test_set_, submit_df, i)
        print('=====================================')


"""
train_set, test_set = train_test_split(train_set, test_size=0.2, random_state=2022)
train_set = pd.concat([train_set, train_set[train_set.label==0]])
cols = ['sn', 'fault_time', 'servertype', 'feature', 'label']
train_data_ = MyDataSet3(train_set[cols])
test_data_ = MyDataSet3(test_set[cols], mode='test')

def lr_lambda(epoch):
    if epoch > 35:
        return 0.02
    if epoch > 20:
        return 0.1
    else:
        return 1

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)
#criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 2.0, 1.0, 1.0]))
criterion = SCELoss()
train_data = iter(train_data_)
test_data = iter(test_data_)
best_f1 = 0
for epoch in range(45):
    running_loss = 0
    for step in range(train_data_.step_max):
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
    # epoch 结束后 evaluate
    preds = []
    with torch.no_grad():
        for step in range(test_data_.step_max):
            model.eval()
            feat, label = next(test_data)
            pred = model(feat).argmax(dim=-1).numpy()
            preds.extend(pred)
    test_set['pred'] = preds
    macro_F1 =  macro_f1(test_set)
    print(f"macro F1: {macro_F1}")
    scheduler.step(macro_F1)
"""