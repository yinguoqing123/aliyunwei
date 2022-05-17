# 微调序列格式
import datetime
from pyexpat import model
import torch
from torch.nn import functional as F
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from torchtext.vocab import vocab
import json
from torch.optim.lr_scheduler import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, default='offline', help="use testA pesu label or not")  # testA伪标签
args = parser.parse_args()

path = "../../data/"

deleted_sn = ['SERVER_13006', 'SERVER_20235', 'SERVER_13175', 'SERVER_3805', 'SERVER_12330', 
              'SERVER_15089', 'SERVER_16241']

crashdump_log_train = pd.read_csv(path + "train_data/preliminary_crashdump_dataset.csv")
crashdump_log_train.columns = ['sn', 'crashdump_time', 'fault_code']
# crashdump_log_test_finala = pd.read_csv('/tcdata/final_crashdump_dataset_b.csv')
# crashdump_log_test_finala.columns = ['sn', 'crashdump_time', 'fault_code']

train_label_df = pd.read_csv(path + "train_data/preliminary_train_label_dataset.csv")
train_label_dfs = pd.read_csv(path + "train_data/preliminary_train_label_dataset_s.csv")
train_label = pd.concat([train_label_df, train_label_dfs])
train_label = train_label[~train_label.sn.isin(deleted_sn)]

test_df_a = pd.read_csv(path + "test_ab/preliminary_submit_dataset_a.csv")
test_df_b = pd.read_csv(path + "test_ab/preliminary_submit_dataset_b.csv")
# test_df_finala = pd.read_csv("/tcdata/final_submit_dataset_b.csv")

crashdump_log_train = crashdump_log_train.merge(train_label, on='sn', how='right')
# crashdump_log_test_finala = crashdump_log_test_finala.merge(test_df_finala, on='sn', how='right')

crashdump_dict = {'pad': 0, 'unk': 1}
for code in list(crashdump_log_train[crashdump_log_train.fault_code.notnull()].fault_code):
    for w in code.split('.'):
        crashdump_dict[w] = crashdump_dict.get(w, len(crashdump_dict))
    
json.dump(crashdump_dict, open("../../tmp_data/crashdump_dict.json", 'w'))
 
def crashdump_process(fault_time, crashdump_time, fault_code):
    if crashdump_time is np.nan:
        return json.dumps([0]*2)
    fault_time = datetime.datetime.strptime(fault_time, '%Y-%m-%d %H:%M:%S')
    crashdump_time = datetime.datetime.strptime(crashdump_time, '%Y-%m-%d %H:%M:%S')
    if not crashdump_time:
        return json.dumps([0]*2)
    if not fault_time - datetime.timedelta(hours=12) < crashdump_time < fault_time + datetime.timedelta(hours=5):
        return json.dumps([0]*2)
    code = [crashdump_dict.get(w, 1) for w in fault_code.split('.')]
    code = [code[1], code[-1]]
    #return json.dumps([crashdump_dict.get(w, 1) for w in fault_code.split('.')])
    return code

crashdump_log_train['crashdump_feature'] = crashdump_log_train.apply(lambda x: crashdump_process(x.fault_time, x.crashdump_time, 
                                                                                 x.fault_code), axis=1)

crashdump_log_train = crashdump_log_train.drop_duplicates(subset=['sn', 'fault_time'])

# crashdump_log_test_finala['crashdump_feature'] = crashdump_log_test_finala.apply(lambda x: crashdump_process(x.fault_time, x.crashdump_time, x.fault_code), axis=1)
# crashdump_log_test_finala = crashdump_log_test_finala.drop_duplicates(subset=['sn', 'fault_time'])

test_df_a['crashdump_feature'] = json.dumps([0]*2)
test_df_b['crashdump_feature'] = json.dumps([0]*2)

cols = ['sn', 'fault_time', 'crashdump_feature']
crashdump_log_train[cols].to_csv('../../tmp_data/crashdump_feature_train.csv', index=False)
# crashdump_log_test_finala[cols].to_csv("../../tmp_data/crashdump_feature_finala.csv", index=False)
test_df_a[cols].to_csv("../../tmp_data/crashdump_feature_test_a.csv", index=False)
test_df_b[cols].to_csv("../../tmp_data/crashdump_feature_test_b.csv", index=False)

           