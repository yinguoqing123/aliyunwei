# dgcnn
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
from data2 import MyDataSet2, macro_f1

train_set = pd.read_csv('../data/train_set2.csv')
submit_df = pd.read_csv('../data/submit_df2.csv')

train_set, test_set = train_test_split(train_set, test_size=0.2, random_state=2022)
cols = ['sn', 'fault_time', 'server_model', 'feature', 'label']
train_data = MyDataSet2(train_set[cols])
test_data = MyDataSet2(test_set[cols])
lookup1 = torch.load('../data/word2idx2.pk')

class Model(nn.Module):
    def __init__(self, in_features=32) -> None:
        super().__init__()
        self.emb1 = nn.Embedding(len(lookup1), 32, padding_idx=0)
        self.block1 = Block(in_features)
        self.block2 = Block(in_features)
        self.classify = nn.Linear(32, 4)
    def forward(self, input):
        x0, x1, mask1, mask2 = input   # x0: [batch_size, sentence_num, word_num]
        x0 = self.emb1(x0)  # [batch_size, sentence_num, word_num, emb]
        b, s, w, d = x0.shape
        x0 = x0.reshape(b*s, w, d)  # b*s, w, d 
        mask1 = mask1.reshape(b*s, -1, 1)  # b*s, w, 1
        x0 = self.block1((x0, mask1))  # b*s, w, emd(out_channel)
        x0 = x0.reshape(b, s, -1)  # b*s, emd  ==> b, s, emd
        mask2 = mask2.reshape(b, s, -1)  
        x0 = self.block2((x0, mask2))  # b, emb
        x0 = F.softmax(self.classify(x0), dim=-1)
        return x0
        
        
class Block(nn.Modelu):
    def __init__(self, in_features=32) -> None:
        super().__init__()
        self.conv1d1 = DilatedGatedConv1D(in_features, in_features, dilation=1, drop_gate=0.1)
        self.conv1d2 = DilatedGatedConv1D(in_features, in_features, dilation=2, drop_gate=0.1)
        self.conv1d3 = DilatedGatedConv1D(in_features, in_features, dilation=4, drop_gate=0.1)
        self.conv1d4 = DilatedGatedConv1D(in_features, in_features, dilation=1, drop_gate=0.1) 
        self.att = AttentionPooling1D(in_features) 
    def forward(self, input):
        x0, mask = input
        x0 = x0.permute(0, 2, 1)
        mask = mask.permute(0, 2, 1)
        x0 = self.conv1d1((x0, mask))
        x0 = self.conv1d2((x0, mask))
        x0 = self.conv1d3((x0, mask))
        x0 = self.conv1d4((x0, mask))
        x0 = x0.permute(0, 2, 1) # 
        mask = mask.permute(0, 2, 1)
        x0 = self.att((x0, mask))
        return x0  # b*emd
        
        
class DilatedGatedConv1D(nn.Module):
    """膨胀门卷积(DGCNN)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation=1,
                 skip_connect=True,
                 drop_gate=None,
                 **kwargs):
        super(DilatedGatedConv1D, self).__init__(**kwargs)
        self.in_channels = in_channels  #词向量的维度
        self.out_channels = out_channels # 卷积后词向量的维度
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.skip_connect = skip_connect
        self.drop_gate = drop_gate
        self.conv1d = nn.Conv1d(
            self.in_channels,
            self.out_channels*2,
            self.kernel_size,
            dilatione=self.dilation,
            padding='same'
        )
        if self.skip_connect and self.in_channels != self.out_channels:
            self.conv1d_1x1 = nn.Conv1d(self.in_channels, self.out_channels, padding='same')
        if self.drop_gate:
            self.dropout = nn.Dropout(drop_gate)
    def forward(self, inputs):
        xo, mask = inputs
        x = xo * mask 
        x = self.conv1d(x)  # batch_size*sentence_num,  out_channels, word_num
        x, g = x[:, :self.out_channels, ...], x[:, self.out_channels:, ...]
        if self.drop_gate:
            g = self.dropout(g)
        g = F.sigmoid(g)
        if self.skip_connect:
            if self.in_channels != self.out_channels:
                xo = self.conv1d_1x1(xo)
            return xo * (1 - g) + x * g
        else:
            return x * g * mask

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
        x = self.k_dense(xo)
        x = self.o_dense(F.tanh(x))  # N, w, 1
        x = x - (1 - mask) * 1e12
        x = F.softmax(x, dim=-2)  # N, w, 1
        return torch.sum(x * xo, dim=-2) # N*emd


model = Model()
criterion = nn.CrossEntropyLoss(weight=torch.tensor([3.0, 2.0, 1.0, 1.0]))
best_f1 = 0
def lr_lambda(epoch):
    if epoch > 15:
        return 0.1
    else:
        return 1

optimizer = optim.Adam(model.parameters(), lr=1e-3)
lr_sche = LambdaLR(optimizer, lr_lambda)

train_data = MyDataSet2(train_set)
train_data_iter = iter(train_data)
test_data = MyDataSet2(test_set)
test_data_iter = iter(test_data)


for epoch in range(20):
    model.train()
    running_loss = 0.0
    for step in range(train_data.step_max):
        feat, label = next(train_data_iter)
        pred = model(feat)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if step % 50 == 49:
            print(f"Epoch {epoch+1}, step {step+1}: {running_loss}")
            running_loss = 0 
    # evaluate
    preds = []
    with torch.no_grad():
        for step in range(test_data.step_max):
            model.eval()
            feat, label = next(test_data_iter)
            pred = model(feat).numpy()
            pred = [json.dumps(p.tolist()) for p  in pred]
            preds.extend(pred)
    test_set['pred'] = preds
    macro_F1 =  macro_f1(test_set)
    name = 0
    if macro_F1 > best_f1:
        torch.save(model.state_dict(), f'../model2/model_{name}.pt')
    test_set.to_csv(f"../valid2/pred_{name}.csv", index=False)
    print(f"macro F1: {macro_F1}")
    
        
