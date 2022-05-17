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

venus_log_train = pd.read_csv(path + "train_data/preliminary_venus_dataset.csv")
venus_log_train.columns = ['sn', 'venus_time', 'module_cause', 'module']
# venus_log_test_finala = pd.read_csv('/tcdata/final_venus_dataset_b.csv')
# venus_log_test_finala.columns = ['sn', 'venus_time', 'module_cause', 'module']

train_label_df = pd.read_csv(path + "train_data/preliminary_train_label_dataset.csv")
train_label_dfs = pd.read_csv(path + "train_data/preliminary_train_label_dataset_s.csv")
train_label = pd.concat([train_label_df, train_label_dfs])
train_label = train_label[~train_label.sn.isin(deleted_sn)]

test_df_a = pd.read_csv(path + "test_ab/preliminary_submit_dataset_a.csv")
test_df_b = pd.read_csv(path + "test_ab/preliminary_submit_dataset_b.csv")
# test_df_finala = pd.read_csv("/tcdata/final_submit_dataset_b.csv")

venus_log_train = venus_log_train.merge(train_label, on='sn', how='right')
# venus_log_test_finala = venus_log_test_finala.merge(test_df_finala, on='sn', how='right')

venus_dict = {'pad': 0, 'unk': 1}
for venus in list(venus_log_train[venus_log_train.module_cause.notnull()].module_cause):
    venus_split = venus.split(',')
    for w in venus_split:
        venus_dict[w] = venus_dict.get(w, len(venus_dict))
        
json.dump(venus_dict, open("../../tmp_data/venus_dict.json", 'w'))
 
def venus_process(fault_time, venus_time, module, module_cause):
    if venus_time is np.nan:
        return json.dumps([[0] * 3])
    
    fault_time = datetime.datetime.strptime(fault_time, '%Y-%m-%d %H:%M:%S')
    venus_time = datetime.datetime.strptime(venus_time, '%Y-%m-%d %H:%M:%S')
    if not fault_time - datetime.timedelta(hours=12) < venus_time < fault_time + datetime.timedelta(hours=1) :
        return json.dumps([[0] * 3])
    
    venus_seq = []
    module_cause = module_cause.split(',')
    for modu in module.split(','):
        index = module_cause.index(modu)
        if 'module' in modu:
            log = module_cause[index: index+3]
            log = [venus_dict.get(l, 1) for l in log]
        else:
            log = module_cause[index: index+2]
            log = [venus_dict.get(l, 1) for l in log] + [0,]
        venus_seq.append(log) 
    return  json.dumps(venus_seq[:4])


venus_log_train['venus_feature'] = venus_log_train.apply(lambda x: venus_process(x.fault_time, x.venus_time, 
                                                                                 x.module, x.module_cause), axis=1)

venus_log_train = venus_log_train.drop_duplicates(subset=['sn', 'fault_time'])

# venus_log_test_finala['venus_feature'] = venus_log_test_finala.apply(lambda x: venus_process(x.fault_time, x.venus_time, 
#                                                                                  x.module, x.module_cause), axis=1)
# venus_log_test_finala = venus_log_test_finala.drop_duplicates(subset=['sn', 'fault_time'])

test_df_a['venus_feature'] = json.dumps([[0]*3])
test_df_b['venus_feature'] = json.dumps([[0]*3])

cols = ['sn', 'fault_time', 'venus_feature']
venus_log_train[cols].to_csv('../../tmp_data/venus_feature_train.csv', index=False)
# venus_log_test_finala[cols].to_csv("../../tmp_data/venus_feature_finala.csv", index=False)
test_df_a[cols].to_csv("../../tmp_data/venus_feature_test_a.csv", index=False)
test_df_b[cols].to_csv("../../tmp_data/venus_feature_test_b.csv", index=False)

           
