from typing import Sequence
import datetime
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

def  macro_f1(overall_df):
    """
    计算得分
    :param target_df: [sn,fault_time,label]
    :param submit_df: [sn,fault_time,label]
    :return:
    """

    weights =  [3/7,  2/7,  1/7,  1/7]

    macro_F1 =  0.
    for i in  range(len(weights)):
        TP =  len(overall_df[(overall_df['label'] == i) & (overall_df['pred'] == i)])
        FP =  len(overall_df[(overall_df['label'] != i) & (overall_df['pred'] == i)])	
        FN =  len(overall_df[(overall_df['label'] == i) & (overall_df['pred'] != i)])
        precision = TP /  (TP + FP)  if  (TP + FP)  >  0  else  0
        recall = TP /  (TP + FN)  if  (TP + FN)  >  0  else  0
        F1 =  2  * precision * recall /  (precision + recall)  if  (precision + recall)  >  0  else  0
        macro_F1 += weights[i]  * F1
        print(f"class {i}, precision: {precision}, recall: {recall}, F1: {F1}")
    return macro_F1  
    
class MyDataSet():
    def __init__(self, data, batch_size=64, mode='train'):
        self.data = data
        self.step = 0
        self.batch_size = batch_size
        self.mode = mode
        self.step_max = (len(self.data) + self.batch_size - 1)//self.batch_size
    def __iter__(self):
        while True:
            start = self.step * self.batch_size
            end = min(self.data.shape[0], start + self.batch_size)
            msg_feat, length_feat1, length_feat2 = [], [], []
            sentence_max, word_max = 0, 0
            for sample in list(self.data.iloc[start:end].feature):
                msg_seq, length_seq, length = json.loads(sample)
                msg = pad_sequence([torch.tensor(msg) for msg in msg_seq], batch_first=True)   # sentence_num * word_num
                msg_feat.append(msg)
                length_feat1.append(torch.tensor(length_seq))
                length_feat2.append(length)
                sentence_max = max(sentence_max, length)
                word_max = max(word_max, msg.shape[1])
        
            batch = np.array([F.pad(sample, (0, word_max-sample.shape[1], 0, sentence_max-sample.shape[0])).numpy() for sample in msg_feat])
            batch = torch.tensor(batch)  # batch_size * sentence_max * word_max
            length_feat1 = pad_sequence(length_feat1, batch_first=True)
            length_feat2 = torch.tensor(length_feat2)
            servertypes = torch.tensor(list(self.data.iloc[start:end].servertype))
            if end == self.data.shape[0]:
                self.step = 0
            else:
                self.step += 1
            if self.mode != 'predict':
                labels = torch.tensor(list(self.data.iloc[start:end].label))
                yield  (batch, servertypes, length_feat1, length_feat2), labels
            else:
                yield  (batch, servertypes, length_feat1, length_feat2)
    def __len__(self):
        return len(self.data) 
    
class MyDataSet2():
    def __init__(self, data, batch_size=64, mode='train'):
        self.data = data
        self.step = 0
        self.batch_size = batch_size
        self.mode = mode
        self.step_max = (len(self.data) + self.batch_size - 1)//self.batch_size
    def __iter__(self):
        while True:
            start = self.step * self.batch_size
            end = min(self.data.shape[0], start + self.batch_size)
            msg_feat, mask1, mask2 = [], [], []
            sentence_max, word_max = 0, 0
            for sample in list(self.data.iloc[start:end].feature):
                msg_seq, length_seq,  length = json.loads(sample)
                msg = pad_sequence([torch.tensor(msg) for msg in msg_seq], batch_first=True)   # sentence_num * word_num
                msg_feat.append(msg)
                mask1 = pad_sequence([torch.ones(length) for length in length_seq], batch_first=True)
                mask1.append(mask1)
                mask2 = torch.ones(length)
                mask2.append(mask2)
                sentence_max = max(sentence_max, length)
                word_max = max(word_max, msg.shape[1])
        
            batch = np.array([F.pad(sample, (0, word_max-sample.shape[1], 0, sentence_max-sample.shape[0])).numpy() for sample in msg_feat])
            batch = torch.tensor(batch)  # batch_size * sentence_max * word_max
            mask1 = np.array([F.pad(sample, (0, word_max-sample.shape[1], 0, sentence_max-sample.shape[0])).numpy() for sample in mask1])
            mask1 = torch.tensor(mask1)
            mask2 = pad_sequence(mask2, batch_first=True)
            servertypes = torch.tensor(list(self.data.iloc[start:end].servertype))
            if end == self.data.shape[0]:
                self.step = 0
            else:
                self.step += 1
            if self.mode != 'predict':
                labels = torch.tensor(list(self.data.iloc[start:end].label))
                yield  (batch, servertypes, mask1, mask2), labels
            else:
                yield  (batch, servertypes, mask1, mask2)
    def __len__(self):
        return len(self.data) 

class MyDataSet3():
    def __init__(self, data, batch_size=64, mode='train'):
        self.data = data
        self.step = 0
        self.batch_size = batch_size
        self.mode = mode
        self.step_max = (len(self.data) + self.batch_size - 1)//self.batch_size
    def __iter__(self):
        while True:
            start = self.step * self.batch_size
            end = min(self.data.shape[0], start + self.batch_size)
            msg_batch, length_seq = [], []
            sentence_max = 0
            for sample in list(self.data.iloc[start:end].feature):
                msgs = json.loads(sample)   # 二维数组 sentence_num * 3 
                length = len(msgs)
                msg_batch.append(torch.tensor(msgs))
                length_seq.append(length)
                sentence_max = max(sentence_max, length)
            
            msg_batch = np.array([F.pad(sample, (0, 0, 0, sentence_max-sample.shape[0])).numpy() for sample in msg_batch])
            msg_batch = torch.tensor(msg_batch)  # batch_size * sentence_max * word_max
            length_seq = torch.tensor(length_seq)
            servertypes = torch.tensor(list(self.data.iloc[start:end].servertype))
            if self.mode != 'predict':
                labels = torch.tensor(list(self.data.iloc[start:end].label))
                yield  (msg_batch, servertypes, length_seq), labels
            else:
                yield  (msg_batch, servertypes, length_seq)
            if end == self.data.shape[0]:
                self.step = 0
                # if self.mode == 'train':
                #     self.data = self.data.sample(frac=1)
            else:
                self.step += 1
    def __len__(self):
        return len(self.data)