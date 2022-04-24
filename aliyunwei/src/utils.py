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
            msg_batch, length_seq, mask = [], [], []
            sentence_max = 0
            for sample in list(self.data.iloc[start:end].feature):
                msgs = json.loads(sample)   # 二维数组 sentence_num * 3 
                length = len(msgs)
                msg_batch.append(torch.tensor(msgs))
                length_seq.append(length)
                mask.append(torch.ones(length))
                sentence_max = max(sentence_max, length)
            
            msg_batch = np.array([F.pad(sample, (0, 0, 0, sentence_max-sample.shape[0])).numpy() for sample in msg_batch])
            msg_batch = torch.tensor(msg_batch)  # batch_size * sentence_max * word_max
            length_seq = torch.tensor(length_seq)
            mask = pad_sequence(mask, batch_first=True)
            servertypes = torch.tensor(list(self.data.iloc[start:end].servertype))
            if self.mode != 'predict':
                labels = torch.tensor(list(self.data.iloc[start:end].label))
                yield  (msg_batch, servertypes, length_seq, mask), labels
            else:
                yield  (msg_batch, servertypes, length_seq, mask)
            if end == self.data.shape[0]:
                self.step = 0
                if self.mode == 'train':
                    self.data = self.data.sample(frac=1)
            else:
                self.step += 1
    def __len__(self):
        return len(self.data)
    
class MyDataSet4():
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
            msg_batch, length_seq, mask = [], [], []
            sentence_max = 0
            for sample in list(self.data.iloc[start:end].feature):
                msgs = json.loads(sample)   # 二维数组 sentence_num * 3 
                length = len(msgs)
                try:
                    msg_batch.append(torch.tensor(msgs))
                except:
                    print(msgs)
                length_seq.append(length)
                mask.append(torch.ones(length))
                sentence_max = max(sentence_max, length)
            
            msg_batch = np.array([F.pad(sample, (0, 0, 0, sentence_max-sample.shape[0])).numpy() for sample in msg_batch])
            msg_batch = torch.tensor(msg_batch)  # batch_size * sentence_max * word_max
            length_seq = torch.tensor(length_seq)
            mask = pad_sequence(mask, batch_first=True)
            servertypes = torch.tensor(list(self.data.iloc[start:end].servertype))
            if self.mode != 'predict':
                labels = torch.tensor(list(self.data.iloc[start:end].label))
                yield  (msg_batch, servertypes, length_seq, mask), labels
            else:
                yield  (msg_batch, servertypes, length_seq, mask)
            if end == self.data.shape[0]:
                self.step = 0
                if self.mode == 'train':
                    self.data = self.data.sample(frac=1, random_state=2022)
            else:
                self.step += 1
    def __len__(self):
        return len(self.data)

class MyDataSet8():
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
            msg_batch, length_seq, mask = [], [], []
            manfeat_batch = []
            sentence_max = 0
            for sample in list(self.data.iloc[start:end].feature):
                msgs, manfeat = json.loads(sample)   # 二维数组 sentence_num * 3 
                length = len(msgs)
                msg_batch.append(torch.tensor(msgs))
                length_seq.append(length)
                mask.append(torch.ones(length))
                sentence_max = max(sentence_max, length)
                manfeat_batch.append(torch.tensor(manfeat))  # sentence_num *  4
            
            msg_batch = np.array([F.pad(sample, (0, 0, 0, sentence_max-sample.shape[0])).numpy() for sample in msg_batch])
            msg_batch = torch.tensor(msg_batch)  # batch_size * sentence_max * word_max
            length_seq = torch.tensor(length_seq)
            mask = pad_sequence(mask, batch_first=True)
            manfeat_batch = np.array([F.pad(sample, (0, 0, 0, 50-sample.shape[0])).numpy() for sample in manfeat_batch])
            manfeat_batch = torch.tensor(manfeat_batch)
            servertypes = torch.tensor(list(self.data.iloc[start:end].servertype))
            if self.mode != 'predict':
                labels = torch.tensor(list(self.data.iloc[start:end].label))
                yield  (msg_batch, servertypes, length_seq, mask, manfeat_batch), labels
            else:
                yield  (msg_batch, servertypes, length_seq, mask, manfeat_batch)
            if end == self.data.shape[0]:
                self.step = 0
                if self.mode == 'train':
                    self.data = self.data.sample(frac=1, random_state=2022)
            else:
                self.step += 1
    def __len__(self):
        return len(self.data)
    
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, coff1=1, coff2=1, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)
        self.dice  = BinaryDiceLoss(**self.kwargs)
        self.coff1 = coff1
        self.coff2 = coff2

    def forward(self, predict, target):    
        total_loss = 0
        loss2 = self.criterion(predict, target)

        predict = torch.softmax(predict, dim=-1)
        target = F.one_hot(target, num_classes=4)
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = self.dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        return self.coff1 * total_loss/target.shape[1] + loss2 * self.coff2