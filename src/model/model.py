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
from .dataset import MyDataSet
import random
import os
import math

intervalbucketnum, cntbucketnum, durationbucketnum = 101, 23, 12

lookup1 = torch.load("../tmp_data/word2idx.pk")
venus_dict = json.load(open('../tmp_data/venus_dict.json', 'r'))
crashdump_dict = json.load(open('../tmp_data/crashdump_dict.json', 'r'))

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
        x = self.o_dense(torch.tanh(x))  # N, s, 1
        x = x - (1 - mask) * 1e12
        x = F.softmax(x, dim=-2)  # N, w, 1
        return torch.sum(x * xo, dim=-2) # N*emd
    
class ScalaOffset(nn.Module):
    def __init__(self, dim=10) -> None:
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(1, 1, dim))
        self.offset = nn.Parameter(torch.zeros(1, 1, dim))
    
    def forward(self, input):
        assert self.dim == input.shape[-1], "dim 维度不对"
        out = input * self.scale + self.offset
        return out
 
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    '''Returns: [seq_len, d_hid]
    '''
    embeddings_table = torch.zeros(n_position, d_hid)
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    return embeddings_table 
    
class RoPEPositionEncoding(nn.Module):
    """旋转式位置编码: https://kexue.fm/archives/8265
    """
    def __init__(self, max_position, embedding_size):
        super(RoPEPositionEncoding, self).__init__()
        position_embeddings = get_sinusoid_encoding_table(max_position, embedding_size)  # [seq_len, hdsz]
        cos_position = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)
        sin_position = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)
        # register_buffer是为了最外层model.to(device)，不用内部指定device
        self.register_buffer('cos_position', cos_position)
        self.register_buffer('sin_position', sin_position)
    
    def forward(self, qw, seq_dim=-2):
        # 默认最后两个维度为[seq_len, hdsz]
        seq_len = qw.shape[seq_dim]
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
        return qw * self.cos_position[:seq_len] + qw2 * self.sin_position[:seq_len]
        
class GateAttentionUnit(nn.Module):
    def __init__(self, unit=46, key_size=16) -> None:
        super().__init__()
        self.unit = unit
        self.key_size = key_size
        self.i_dense = nn.Linear(unit, 2*unit+key_size)
        self.o_dense = nn.Linear(unit, unit)
        self.scaleoffset_q = ScalaOffset(dim=key_size)
        self.scaleoffset_k = ScalaOffset(dim=key_size)
        self.query = nn.Linear(unit, 1)
        self.rope = RoPEPositionEncoding(50, 16)
        
    def forward(self, input):
        x, mask = input
        mask_trans = mask[:, None, :]
        u, v, qk = torch.split(F.silu(self.i_dense(x)), [self.unit, self.unit, self.key_size], dim=-1)
        q, k = self.scaleoffset_q(qk), self.scaleoffset_k(qk)
        q, k = self.rope(q), self.rope(k)
        A = torch.matmul(q, k.transpose(1, 2)) / (self.key_size**0.5)
        A = torch.softmax(A * mask_trans + (1-mask_trans) * -1e12, dim=-1)
        # out = torch.matmul(A, v) * u
        out = torch.matmul(A, v) 
        # out = self.o_dense(out)
        prob = self.query(out).squeeze()  # 
        prob = prob * mask - (1 - mask) * 1e12
        prob = torch.softmax(prob, dim=-1)
        return torch.sum(out*prob.unsqueeze(dim=-1), dim=1)
        
        
class MyModel(nn.Module):
    def __init__(self, att_cate='pool') -> None:
        super(MyModel, self).__init__()
        self.emb_msg = nn.Embedding(len(lookup1), 12, padding_idx=0)
        self.emb_servermodel = nn.Embedding(88, 10, padding_idx=0)
        self.emb_msginterl = nn.Embedding(intervalbucketnum, 5)
        self.emb_msgcnt = nn.Embedding(cntbucketnum, 3)
        self.emb_msgduration = nn.Embedding(durationbucketnum, 2)
        self.emb_venus = nn.Embedding(len(venus_dict), 5, padding_idx=0)
        self.emb_crashdump = nn.Embedding(len(crashdump_dict), 3, padding_idx=0)
        if att_cate == 'gate':
            self.att = GateAttentionUnit(46)
        else:
            self.att = AttentionPooling1D(46)
            
        self.classify = nn.Linear(46+10+20+12, 4)
        # self.classify = nn.Linear(46+10)
        # self.classify.bias.data = torch.tensor([-2.38883658, -1.57741002, -0.57731536, -1.96360971])
  
    def forward(self, feat):
        msg_batch, msg_mask, venus_batch, venus_mask, server_model, crashdump = feat  # len1 batch_size * sentence_num
        
        msgs, msg_interval, msg_cnt, msg_duration = msg_batch[..., :3], msg_batch[..., 3], msg_batch[..., 4], msg_batch[..., 5]  # (b, s, 3), (b, s) (b, s) (b, s)
        msgs = self.emb_msg(msgs)  # (b, s, 3, d)
        msg_interval = self.emb_msginterl(msg_interval)
        msg_cnt = self.emb_msgcnt(msg_cnt)
        msg_duration = self.emb_msgduration(msg_duration)
        b, s, w, d = msgs.shape
        word_emb = msgs.reshape(b, s, w*d)
        word_emb = torch.concat([word_emb, msg_interval, msg_cnt, msg_duration], dim=-1)
        att_emb = self.att((word_emb, msg_mask))
        
        venus = self.emb_venus(venus_batch)
        b, s, w, d = venus.shape
        venus = venus.reshape(b, s, w*d) # bsz, s, 4, dim
        venus = torch.sum(venus, dim=1) / torch.sum(venus_mask, dim=-1, keepdim=True)
        
        server_model = self.emb_servermodel(server_model)
        
        crashdump = self.emb_crashdump(crashdump) # b, 4, dim
        crashdump = crashdump.view(b, -1)

        score = self.classify(torch.concat([att_emb, venus, server_model, crashdump], dim=-1))
        # score = self.classify(torch.concat([att_emb, server_model], dim=-1))
        return score
    
