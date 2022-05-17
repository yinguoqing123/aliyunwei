import pandas as pd

deleted_sn = ['SERVER_13006', 'SERVER_20235', 'SERVER_13175', 'SERVER_3805', 'SERVER_12330', 
              'SERVER_15089', 'SERVER_16241']

train_msg_feature = pd.read_csv('../../tmp_data/msg_feature_train.csv')
train_venus_feature = pd.read_csv("../../tmp_data/venus_feature_train.csv")
train_crashdump_feature = pd.read_csv("../../tmp_data/crashdump_feature_train.csv")

train_label = pd.read_csv('../../data/train_data/preliminary_train_label_dataset.csv')
train_label_s = pd.read_csv('../../data/train_data/preliminary_train_label_dataset_s.csv')
train_label = pd.concat([train_label, train_label_s])
train_label = train_label[~train_label.sn.isin(deleted_sn)]

train_set = train_label.merge(train_msg_feature, on=['sn', 'fault_time'], how='left') \
    .merge(train_venus_feature, on=['sn', 'fault_time'], how='left') \
    .merge(train_crashdump_feature, on=['sn', 'fault_time'], how='left')
    
train_set.to_csv("../../tmp_data/train_set.csv", index=False)



train_msg_feature = pd.read_csv('../../tmp_data/msg_feature_test_a.csv')
train_venus_feature = pd.read_csv("../../tmp_data/venus_feature_test_a.csv")
train_crashdump_feature = pd.read_csv("../../tmp_data/crashdump_feature_test_a.csv")

train_label = pd.read_csv('../../data/test_ab/preliminary_submit_dataset_a.csv')

train_set = train_label.merge(train_msg_feature, on=['sn', 'fault_time'], how='left') \
    .merge(train_venus_feature, on=['sn', 'fault_time'], how='left') \
    .merge(train_crashdump_feature, on=['sn', 'fault_time'], how='left')
    
train_set.to_csv("../../tmp_data/test_set_a.csv", index=False)



train_msg_feature = pd.read_csv('../../tmp_data/msg_feature_test_b.csv')
train_venus_feature = pd.read_csv("../../tmp_data/venus_feature_test_b.csv")
train_crashdump_feature = pd.read_csv("../../tmp_data/crashdump_feature_test_b.csv")

train_label = pd.read_csv('../../data/test_ab/preliminary_submit_dataset_b.csv')

train_set = train_label.merge(train_msg_feature, on=['sn', 'fault_time'], how='left') \
    .merge(train_venus_feature, on=['sn', 'fault_time'], how='left') \
    .merge(train_crashdump_feature, on=['sn', 'fault_time'], how='left')
    
train_set.to_csv("../../tmp_data/test_set_b.csv", index=False)

# train_msg_feature = pd.read_csv('../../tmp_data/msg_feature_finala.csv')
# train_venus_feature = pd.read_csv("../../tmp_data/venus_feature_finala.csv")
# train_crashdump_feature = pd.read_csv("../../tmp_data/crashdump_feature_finala.csv")

# train_label = pd.read_csv('/tcdata/final_submit_dataset_a.csv')

# train_set = train_label.merge(train_msg_feature, on=['sn', 'fault_time'], how='left') \
#     .merge(train_venus_feature, on=['sn', 'fault_time'], how='left') \
#     .merge(train_crashdump_feature, on=['sn', 'fault_time'], how='left')
    
# train_set.to_csv("../../tmp_data/finala_set.csv", index=False)