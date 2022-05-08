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
# msg_log_test_finala = pd.read_cvs("/tcdata/final_sel_log_dataset_a.csv")

train_label_df = pd.read_csv(path + "train_data/preliminary_train_label_dataset.csv")
train_label_dfs = pd.read_csv(path + "train_data/preliminary_train_label_dataset_s.csv")
train_label = pd.concat([train_label_df, train_label_dfs])
train_label = train_label[~train_label.sn.isin(deleted_sn)]
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
# msg_log_test_finala = msg_log_test_finala = groupby('sn').agg({'time': list, 'msg': list, 'server_model': max}).reset_index()
msg_log_train.columns = ['sn', 'time_seq', 'msg_seq', 'server_model']
msg_log_test_a.columns = ['sn', 'time_seq', 'msg_seq', 'server_model']
msg_log_test_b.columns = ['sn', 'time_seq', 'msg_seq', 'server_model']
msg_feature_train = msg_log_train.merge(train_label, on='sn', how='right')
msg_feature_test_a = msg_log_test_a.merge(test_df_a, on='sn', how='right')
msg_feature_test_b = msg_log_test_b.merge(test_df_b, on='sn', how='right')
# msg_log_test_finala = msg_log_test_finala.merge(test_df_finala, on='sn', how='right')


#  =======================  特征处理  ============================

intervalbucket = [2.0, 21.0, 42.0, 64.0, 89.0, 116.0, 145.47999999999956, 176.0, 208.0, 238.0, 266.0, 293.0, 327.0, 362.0400000000009, 402.0, 444.0, 491.0, 539.0, 579.0, 618.0, 661.0, 716.0, 772.0, 835.0, 894.0, 953.0, 1016.0, 1076.0, 1133.0, 1191.0, 1245.0, 1312.0, 1376.0, 1436.0, 1500.0, 1556.0, 1619.0, 1680.0, 1744.0, 1800.0, 1873.0, 1982.0, 2062.0, 2156.0, 2250.0, 2368.5999999999985, 2501.0, 2617.0, 2745.0, 2862.0, 2995.0, 3119.0, 3252.0, 3390.0, 3504.0, 3590.0, 3655.0, 3715.0, 3767.0, 3817.0, 3857.0, 3900.0, 3949.0, 4002.0, 4060.0, 4140.0, 4230.0, 4329.0, 4451.440000000002, 4620.0, 4823.0, 5052.679999999993, 5295.0, 5532.0, 5770.0, 6040.0, 6334.0, 6605.1600000000035, 6883.0, 7201.320000000007, 7697.0, 8358.48000000001, 9096.559999999998, 10086.0, 11147.0, 12345.800000000003, 13803.0, 15678.920000000013, 17796.079999999987, 20064.359999999986, 22474.0, 24988.0, 27292.0, 28895.0, 30239.51999999999, 32214.0, 34237.71999999997, 38868.76000000001, 46575.0, 56198.12000000046]
cntbucket = [1.0, 2.0, 3.0, 4.0, 5., 6., 7, 8, 9, 10, 12, 14, 17, 20, 25, 30, 35, 45, 60, 90, 150, 200]
durationbucket = [1.0, 3.0, 10., 40, 100, 500, 1000, 5000, 10000, 30000, 70000]

def getbucket(x, bucket):
    for i in range(len(bucket)):
        if x < bucket[i]:
            return i
    return len(bucket)

def seqfilter(time_seq, msg_seq, time):
    ret_msg_seq = []
    time_msg_seq = list(zip(time_seq, msg_seq))
    time_msg_seq.sort(key=lambda x: x[0])
    time = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    time_minus24h = time - datetime.timedelta(hours=48)
    msgdict = defaultdict(lambda :[])
    for cur_time, msg in time_msg_seq:
        cur_time = datetime.datetime.strptime(cur_time, '%Y-%m-%d %H:%M:%S')
        if cur_time > time_minus24h and cur_time<time:
            msgdict[msg] = msgdict[msg] + [cur_time]
        
    for msg in msgdict:
        msg_process = [word.strip().lower() for word in msg.strip().split('|')][:3]
        msg_process += (3-len(msg_process)) * ['pad']
        msg_process = lookup1(msg_process)
        curtime_seq = msgdict[msg]
        curtime_seq.sort()
        pretime = curtime_seq[0]
        frstInterval = (time - pretime).total_seconds()
        cnt = 1
        duration = 0
        for curtime in curtime_seq[1:]:
            if (curtime - pretime).total_seconds() <  5 * 3600:
                cnt += 1
                duration = (curtime - pretime).total_seconds()
            else:
                frstInterval = getbucket(frstInterval, intervalbucket)
                cnt = getbucket(cnt, cntbucket)
                duration = getbucket(duration, durationbucket)
                ret_msg_seq.append((msg_process + [frstInterval, cnt, duration], pretime))
                pretime = curtime
                frstInterval = (time - pretime).total_seconds()
                cnt = 1
                duration = 0
        frstInterval = getbucket(frstInterval, intervalbucket)
        cnt = getbucket(cnt, cntbucket)
        duration = getbucket(duration, durationbucket)
        ret_msg_seq.append((msg_process + [frstInterval, cnt, duration], pretime))
    if not ret_msg_seq:
        return json.dumps([lookup1(3*['pad']) + [0, 0, 0]])
    ret_msg_seq.sort(key=lambda x: x[1])
    ret_msg_seq = [ret_msg[0] for ret_msg in ret_msg_seq]
    return json.dumps(ret_msg_seq[-50:])   

msg_feature_train['msg_feature'] = msg_feature_train.apply(lambda x: seqfilter(x.time_seq, x.msg_seq, x.fault_time), axis=1)
msg_feature_test_a['msg_feature'] = msg_feature_test_a.apply(lambda x: seqfilter(x.time_seq, x.msg_seq, x.fault_time), axis=1)
msg_feature_test_b['msg_feature'] = msg_feature_test_b.apply(lambda x: seqfilter(x.time_seq, x.msg_seq, x.fault_time), axis=1)
# msg_feature_test_finala['feature'] = msg_feature_test_finala.apply(lambda x: seqfilter(x.time_seq, x.msg_seq, x.fault_time), axis=1)
msg_feature_train['server_model'] = msg_feature_train.server_model.map(servermodel2id) 
msg_feature_test_a['server_model'] = msg_feature_test_a.server_model.map(servermodel2id) 
msg_feature_test_b['server_model'] = msg_feature_test_b.server_model.map(servermodel2id) 
# msg_feature_test_finala['server_model'] = msg_feature_test_finala.server_model.map(servermodel2id) 

cols = ['sn', 'fault_time', 'msg_feature', 'server_model']
msg_feature_train[cols].to_csv("../../tmp_data/msg_feature_train.csv", index=False)
msg_feature_test_a[cols].to_csv("../../tmp_data/msg_feature_test_a.csv", index=False)
msg_feature_test_b[cols].to_csv("../../tmp_data/msg_feature_test_b.csv", index=False)
# msg_feature_test_finala[cols].to_csv("../../tmp_data/msg_feature_test_finala.csv", index=False)




          