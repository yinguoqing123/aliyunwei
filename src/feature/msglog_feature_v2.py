# 微调序列格式
import datetime
from pyexpat import model
import torch
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from torchtext.vocab import vocab
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import json
from sklearn.model_selection import train_test_split, KFold
from torch.optim.lr_scheduler import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, default='offline', help="use testA pesu label or not")  # testA伪标签
args = parser.parse_args()

path = "../../data/"

deleted_sn = ['SERVER_13006', 'SERVER_20235', 'SERVER_13175', 'SERVER_3805', 'SERVER_12330', 
              'SERVER_15089', 'SERVER_16241']

msg_log_train = pd.read_csv(path + "train_data/preliminary_sel_log_dataset.csv")
msg_log_test_a = pd.read_csv(path + "test_ab/preliminary_sel_log_dataset_a.csv")
msg_log_test_b = pd.read_csv(path + "test_ab/preliminary_sel_log_dataset_b.csv")
# msg_log_test_finala = pd.read_csv("/tcdata/final_sel_log_dataset_a.csv")

train_label_df = pd.read_csv(path + "train_data/preliminary_train_label_dataset.csv")
train_label_dfs = pd.read_csv(path + "train_data/preliminary_train_label_dataset_s.csv")
train_label = pd.concat([train_label_df, train_label_dfs]).drop_duplicates()
# train_label = train_label[~train_label.sn.isin(deleted_sn)]
test_df_a = pd.read_csv(path + "test_ab/preliminary_submit_dataset_a.csv")
test_df_b = pd.read_csv(path + "test_ab/preliminary_submit_dataset_b.csv")
# test_df_finala = pd.read_csv("/tcdata/final_submit_dataset_a.csv")


# 构建词表
wordDict = {}
for msg in list(msg_log_train.msg):
    for word in msg.strip().split('|'):
        word = word.strip()
        wordDict[word.lower()] = wordDict.get(word.lower(), 0) + 1


try:
    lookup1 = torch.load('../../tmp_data/word2idx.pk')
except:
    lookup1 = vocab(wordDict)
    lookup1.insert_token('pad', 0)
    lookup1.insert_token('unk', 1)
    lookup1.set_default_index(1)
    torch.save(lookup1, '../../tmp_data/word2idx.pk')


servermodel2id = dict()
servermodels = sorted(list(set(msg_log_train.server_model)))
for servermodel in servermodels:
    servermodel2id[servermodel] = len(servermodel2id) + 1
    
servermodel2id['pad'] = 0

msg_log_train = msg_log_train.groupby('sn').agg({'time': list, 'msg': list, 'server_model': max}).reset_index()
msg_log_test_a = msg_log_test_a.groupby('sn').agg({'time': list, 'msg': list, 'server_model': max}).reset_index()
msg_log_test_b = msg_log_test_b.groupby('sn').agg({'time': list, 'msg': list, 'server_model': max}).reset_index()
# msg_log_test_finala = msg_log_test_finala.groupby('sn').agg({'time': list, 'msg': list, 'server_model': max}).reset_index()
msg_log_train.columns = ['sn', 'time_seq', 'msg_seq', 'server_model']
msg_log_test_a.columns = ['sn', 'time_seq', 'msg_seq', 'server_model']
msg_log_test_b.columns = ['sn', 'time_seq', 'msg_seq', 'server_model']
# msg_log_test_finala.columns = ['sn', 'time_seq', 'msg_seq', 'server_model']
msg_feature_train = msg_log_train.merge(train_label, on='sn', how='right')
msg_feature_test_a = msg_log_test_a.merge(test_df_a, on='sn', how='right')
msg_feature_test_b = msg_log_test_b.merge(test_df_b, on='sn', how='right')
# msg_feature_test_finala = msg_log_test_finala.merge(test_df_finala, on='sn', how='right')


def seqfilter(time_seq, msg_seq, time):
    ret_msg_seq = []
    time_msg_seq = list(zip(time_seq, msg_seq))
    time_msg_seq.sort(key=lambda x: x[0])
    time = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    time_minus24h = time - datetime.timedelta(hours=72)
    msg_visited = set()
    for cur_time, msg in time_msg_seq:
        cur_time = datetime.datetime.strptime(cur_time, '%Y-%m-%d %H:%M:%S')
        if cur_time > time_minus24h and cur_time < time + datetime.timedelta(hours=12):
            if msg not in msg_visited:
                msg_visited.add(msg)
                msg_process = [word.strip().lower() for word in msg.strip().split('|')][:3]
                msg_process += (3-len(msg_process)) * ['pad']
                msg_process = lookup1(msg_process)
                ret_msg_seq.append(msg_process)

    if not ret_msg_seq:
        return json.dumps([lookup1(3*['pad'])])
    
    return json.dumps(ret_msg_seq[-40:])   

msg_feature_train['msg_feature'] = msg_feature_train.apply(lambda x: seqfilter(x.time_seq, x.msg_seq, x.fault_time), axis=1)
msg_feature_test_a['msg_feature'] = msg_feature_test_a.apply(lambda x: seqfilter(x.time_seq, x.msg_seq, x.fault_time), axis=1)
msg_feature_test_b['msg_feature'] = msg_feature_test_b.apply(lambda x: seqfilter(x.time_seq, x.msg_seq, x.fault_time), axis=1)
# msg_feature_test_finala['msg_feature'] = msg_feature_test_finala.apply(lambda x: seqfilter(x.time_seq, x.msg_seq, x.fault_time), axis=1)
msg_feature_train['server_model'] = msg_feature_train.server_model.map(servermodel2id) 
msg_feature_test_a['server_model'] = msg_feature_test_a.server_model.map(servermodel2id) 
msg_feature_test_b['server_model'] = msg_feature_test_b.server_model.map(servermodel2id) 
# msg_feature_test_finala['server_model'] = msg_feature_test_finala.server_model.map(servermodel2id) 

cols = ['sn', 'fault_time', 'msg_feature', 'server_model']
msg_feature_train[cols].to_csv("../../tmp_data/msg_feature_train.csv", index=False)
msg_feature_test_a[cols].to_csv("../../tmp_data/msg_feature_test_a.csv", index=False)
msg_feature_test_b[cols].to_csv("../../tmp_data/msg_feature_test_b.csv", index=False)
# msg_feature_test_finala[cols].to_csv("../../tmp_data/msg_feature_finala.csv", index=False)




          