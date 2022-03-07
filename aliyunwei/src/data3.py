# 改变分词  数据格式
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

dir = "D:\\ai-risk\\aliyunwei\\data\\"

train_log = pd.read_csv(dir + "preliminary_sel_log_dataset.csv")
train_df = pd.read_csv(dir + 'preliminary_train_label_dataset.csv')
submit_df = pd.read_csv(dir + 'preliminary_submit_dataset_a.csv')
train_log_a = pd.read_csv(dir + "preliminary_sel_log_dataset_a.csv")
train_df_a = pd.read_csv(dir + 'preliminary_train_label_dataset_s.csv')
train_log = pd.concat([train_log, train_log_a])
train_df = pd.concat([train_df, train_df_a])
train_df = train_df.drop_duplicates()

# 构建词表
wordDict = {}
for msg in list(train_log.msg):
    for word in msg.strip().split('|'):
        word = word.strip()
        wordDict[word.lower()] = wordDict.get(word.lower(), 0) + 1
# # 加入server_model
# for server_model in list(train_log.server_model):
#     wordDict[server_model] = wordDict.get(server_model, 0) + 1 

try:
    lookup1 = torch.load('../data/word2idx3.pk')
except:
    lookup1 = vocab(wordDict)
    lookup1.insert_token('pad', 0)
    lookup1.insert_token('unk', 1)
    lookup1.set_default_index(1)
    torch.save(lookup1, '../data/word2idx3.pk')

wordDict = sorted(wordDict.items(), key=lambda x: x[1], reverse=True)
word2id = dict()
for word, freq in wordDict:
    if word not in word2id:
        word2id[word] = len(word2id) + 1
word2id['pad'] = 0
word2id['unk'] = len(word2id)

servertype2id = dict()
servertypes = sorted(list(set(train_log.server_model)))
for servertype in servertypes:
    servertype2id[servertype] = len(servertype2id) + 1
servertype2id['pad'] = 0

train_log_agg = train_log.groupby('sn').agg({'time': list, 'msg': list, 'server_model': max}).reset_index()
train_log_agg.columns = ['sn', 'time_seq', 'msg_seq', 'server_model']
train_df = train_log_agg.merge(train_df, on='sn', how='inner')
submit_df = train_log_agg.merge(submit_df, on='sn', how='inner')

# 数据转换
def seqfilter(time_seq, msg_seq, time):
    ret_msg_seq = []
    time_msg_seq = list(zip(time_seq, msg_seq))
    time_msg_seq.sort(key=lambda x: x[0])
    time = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    for i in range(len(time_msg_seq)):
        cur_time = datetime.datetime.strptime(time_msg_seq[i][0], '%Y-%m-%d %H:%M:%S')
        cur_time_minus24h = cur_time - datetime.timedelta(hours=24)
        if  cur_time_minus24h < cur_time < time:
            msg = time_msg_seq[i][1]
            #if i > 0 and msg == time_msg_seq[i-1][1]:
            #    continue
            msg = [word.strip().lower() for word in msg.strip().split('|')][:3]
            msg += (3-len(msg)) * ['pad'] 
            ret_msg_seq.append(lookup1(msg)) 
    if not ret_msg_seq:
        return None
    return json.dumps((ret_msg_seq[-50:]))

train_df['feature'] = train_df.apply(lambda x: seqfilter(x.time_seq, x.msg_seq, x.fault_time), axis=1)
train_df['servertype'] = train_df.server_model.map(servertype2id)
train_df = train_df[~train_df.feature.isnull()]
train_df.to_csv('../data/train_set3.csv', index=False)
submit_df['feature'] = submit_df.apply(lambda x: seqfilter(x.time_seq, x.msg_seq, x.fault_time), axis=1)
submit_df['servertype'] = submit_df.server_model.map(servertype2id)
submit_df.to_csv("../data/submit_df3.csv", index=False)


    