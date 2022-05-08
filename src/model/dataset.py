import json
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

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
            msg_batch, msg_mask = [], []
            venus_batch, venus_mask = [], []
            msg_sentence_max, venus_sentence_max = 0, 0
            
            for sample in list(self.data.iloc[start:end].msg_feature.values):
                msgs = json.loads(sample)   # 二维数组 sentence_num * 3 
                msg_batch.append(torch.tensor(msgs))
                msg_mask.append(torch.ones(len(msgs)))
                msg_sentence_max = max(msg_sentence_max, len(msgs))
                
            for sample in list(self.data.iloc[start:end].venus_feature.values):
                venus = json.loads(sample)   # 二维数组 sentence_num * 3 
                venus_batch.append(torch.tensor(venus))         
                venus_mask.append(torch.ones(len(venus)))
                venus_sentence_max = max(venus_sentence_max, len(venus))
            
            msg_batch = np.array([F.pad(sample, (0, 0, 0, msg_sentence_max-sample.shape[0])).numpy() for sample in msg_batch])
            venus_batch = np.array([F.pad(sample, (0, 0, 0, venus_sentence_max-sample.shape[0])).numpy() for sample in venus_batch])
            msg_batch = torch.tensor(msg_batch)  # batch_size * sentence_max * word_num
            venus_batch = torch.tensor(venus_batch)
            msg_mask = pad_sequence(msg_mask, batch_first=True)
            venus_mask = pad_sequence(msg_mask, batch_first=True)
            servermodel = torch.tensor(list(self.data.iloc[start:end].server_model))
            crashdump = torch.tensor(list(self.data.iloc[start:end].crashdump_feature))
            if self.mode != 'predict':
                labels = torch.tensor(list(self.data.iloc[start:end].label))
                yield  (msg_batch, msg_mask, venus_batch, venus_mask, servermodel, crashdump), labels
            else:
                yield  (msg_batch, msg_mask, venus_batch, venus_mask, servermodel, crashdump)
            if end == self.data.shape[0]:
                self.step = 0
                if self.mode == 'train':
                    self.data = self.data.sample(frac=1, random_state=2022)
            else:
                self.step += 1
    def __len__(self):
        return len(self.data)
