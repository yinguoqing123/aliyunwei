# 微调序列格式  10分钟统计
import datetime
from email.policy import default
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
from ema import EMA
from torch.optim.lr_scheduler import *
import copy

dir = "D:\\ai-risk\\aliyunwei\\data\\"

train_log = pd.read_csv(dir + "preliminary_sel_log_dataset.csv")
train_df = pd.read_csv(dir + 'preliminary_train_label_dataset.csv')
submit_df = pd.read_csv(dir + 'preliminary_submit_dataset_b.csv')
train_log_a = pd.read_csv(dir + "preliminary_sel_log_dataset_b.csv")
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
    lookup1 = torch.load('../data/word2idx6.pk')
except:
    lookup1 = vocab(wordDict)
    lookup1.insert_token('pad', 0)
    lookup1.insert_token('unk', 1)
    lookup1.set_default_index(1)
    torch.save(lookup1, '../data/word2idx6.pk')

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

# # 人工提取句子统计特征
# sentence2id = {'pad': 0}
# sentenceMapLabel = {0: [], 1:[], 2:[], 3:[]}
# for time_seq, msg_seq,  time, label  in zip(train_df.time_seq, train_df.msg_seq, train_df.fault_time, train_df.label):
#     time = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
#     time_minus24h = time - datetime.timedelta(hours=24)
#     time_msg_seq = list(zip(time_seq, msg_seq))
#     time_msg_seq.sort(key=lambda x: x[0])
#     for t, m in time_msg_seq:
#         t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
#         m = m.strip()
#         if t > time_minus24h and t < time:
#             if m not in sentence2id:
#                 sentence2id[m] = len(sentence2id) 
#             sentenceMapLabel[label].append(sentence2id[m])

# prior_mean = np.array([0.089383, 0.204945, 0.560175, 0.145498])

# class0 = Counter(sentenceMapLabel[0])
# class1 = Counter(sentenceMapLabel[1])
# class2 = Counter(sentenceMapLabel[2])
# class3 = Counter(sentenceMapLabel[3])

# manualfeat = np.zeros((len(sentence2id), 8)) 
# def calFeat(i):
#     classi = Counter(sentenceMapLabel[i])
#     classi_sum = sum(classi.values())
#     for key in classi:
#         classi[key] = classi[key]/classi_sum
    
#     for j in range(len(sentence2id)):
#         manualfeat[j][i] = classi.get(j, prior_mean[i])
        
# # [[0.002565982404692082, 0.002858219952705713, 0.000325893604790636, 0.00013199577613516366, 0.08943399808558346, 0.20498631179153606, 0.5601007832353634, 0.1454799067404615]
# calFeat(0)
# calFeat(1) 
# calFeat(2)
# calFeat(3)       

# for j in range(len(sentence2id)):
#     cnt = class0.get(j, 0) + class1.get(j, 0) + class2.get(j, 0) + class3.get(j, 0) + 1e-12
#     weight = cnt / (cnt + 40)
#     manualfeat[j][4] = weight * class0.get(j, 0) / cnt + (1-weight) * prior_mean[0]
#     manualfeat[j][5] = weight * class1.get(j, 0) / cnt + (1-weight) * prior_mean[1]
#     manualfeat[j][6] = weight * class2.get(j, 0) / cnt + (1-weight) * prior_mean[2]
#     manualfeat[j][7] = weight * class3.get(j, 0) / cnt + (1-weight) * prior_mean[3]
     

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
    manual_feat_seq = []
    time_msg_seq = list(zip(time_seq, msg_seq))
    time_msg_seq.sort(key=lambda x: x[0])
    time = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    time_minus24h = time - datetime.timedelta(hours=24)
    msgdict = defaultdict(lambda :[])
    for cur_time, msg in time_msg_seq:
        cur_time = datetime.datetime.strptime(cur_time, '%Y-%m-%d %H:%M:%S')
        if cur_time > time_minus24h:
            msgdict[msg] = msgdict[msg] + [cur_time]
        
    for msg in msgdict:
        # 人工特征加入
        # id = sentence2id.get(msg.strip(), 0)
        # manfeat = manualfeat[id]
        # manual_feat_seq.append((manfeat, msgdict[msg][0]))
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
            if (curtime - pretime).total_seconds() < 5 * 3600:
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
    # manual_feat_seq.sort(key=lambda x: x[1])
    # manual_feat_seq = [list(manfeat[0]) for manfeat in manual_feat_seq]
    return json.dumps(ret_msg_seq[-50:])       
                

train_df['feature'] = train_df.apply(lambda x: seqfilter(x.time_seq, x.msg_seq, x.fault_time), axis=1)
train_df['servertype'] = train_df.server_model.map(servertype2id)
train_df = train_df[~train_df.feature.isnull()]
train_df.to_csv('../data/train_set7.csv', index=False)
submit_df['feature'] = submit_df.apply(lambda x: seqfilter(x.time_seq, x.msg_seq, x.fault_time), axis=1)
submit_df['servertype'] = submit_df.server_model.map(servertype2id)
submit_df.to_csv("../data/submit_df7.csv", index=False)



            