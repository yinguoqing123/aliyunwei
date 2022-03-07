# -*- encoding: utf-8 -*-
"""
pair-classify
"""

from pickletools import optimize
import pandas as pd
from sklearn.model_selection import learning_curve
import numpy as np
import os
import random
import tensorflow as tf 
import bert4keras
from bert4keras.backend import multilabel_categorical_crossentropy, sparse_multilabel_categorical_crossentropy
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.layers import EfficientGlobalPointer
import logging
os.environ['USE_TF'] = '1'

root = '/kaggle/input/feedback-prize-2021'
root = f'D:\Download\feedback-prize-2021'

from transformers import LongformerModel, RobertaTokenizerFast, RobertaTokenizer
from transformers import create_optimizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

df_label = pd.read_csv(os.path.join(root, 'train.csv'))
label_map = {'Lead': 0, 'Position': 1, 'Evidence': 2, 'Claim': 3, 'Concluding Statement': 4,
       'Counterclaim': 5, 'Rebuttal': 6}
categorys = len(label_map)
df_label['discourse_type'] = df_label.discourse_type.map(label_map)

files = os.listdir(os.path.join(root, 'train'))
random.seed(2022)
random.shuffle(files)
train_files = files[:int(0.9*len(files))]
val_files = files[int(0.9*len(files)):]

def file2Data(files):
    # 数据处理  
    # ---------- paragraph1----------------------------------,  
    # data形式 [[(sentence1, label), (sentence2, label), ....],   ....]
    data = []
    for file in files:
        with open(os.path.join(root, 'train', file)) as f:
            artical = ' '.join([f.read().strip().split('\n\n')])
            texts_labels = df_label[df_label.id == file.split('.')[0]].sort_values(by='discourse_start')[['discourse_text', 'discourse_type', 'predictionstring']]
            texts_labels = list(zip(list(texts_labels.discourse_text),  list(texts_labels.discourse_type)))
    return data

def wordTokenMapping(word, token):
    # word: "i love china. you love shit"
    # token: [(0, 1), (2, 6,)]
    wordmap = []  #[(0, 1), (2, 6), (7, 13)]
    start = 0
    for w in word.split():
        wordmap.append((start, start+len(w)))
        start += len(w) + 1
    i, j = 0, 0
    wordtokenmap = [[] for _ in range(len(word.split()))]   # [[index1, index2]]
    tokenwordmap = [[] for _ in range(len(token))]
    while i < len(word.split()) and j < len(token):
        if wordmap[i][0] <= token[j][0] and wordmap[i][1] >= token[j][1]:
            wordtokenmap[i].append(j)
            tokenwordmap[j].append(i)
        if wordmap[i][1] == token[j][1]:
            i += 1
        j += 1
    return wordtokenmap[:tokenwordmap[-1][-1]+1], tokenwordmap

class DataLoader(DataGenerator):
    def __iter__(self, random=False):
        batch_data, batch_label_origin, articals = [], [], []
        batch_token_ids, batch_masks, batch_offsets = [], [], []
        for is_end, file in self.sample(random=False):  
            with open(os.path.join(root, 'train', file), encoding='utf-8') as f:
                artical = ' '.join(f.read().strip().split())
                articals.append(artical)
                df = df_label[df_label.id == file.split('.')[0]].sort_values(by='discourse_start')[['discourse_text', 'discourse_type', 'predictionstring']]
                label_pos = [s.split() for s in list(df.predictionstring)]
                labels_ = list(df.discourse_type)
                batch_data.append(artical)
                batch_label_origin.append(list(zip(label_pos, labels_)))  # [([1, 2, 3, 4], lead), ([5, 6, 7, 8], conclusion), ...]
                if is_end or len(batch_data) == self.batch_size:
                    for data in batch_data:
                        tokens = tokenizer.encode_plus(data, add_special_tokens=False, return_offsets_mapping=True, 
                                                            padding=True, truncation=True)
                        batch_token_ids.append(tokens['input_ids'])
                        batch_masks.append(tokens['attention_mask'])
                        batch_offsets.append(tokens['offset_mapping'])
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_masks = sequence_padding(batch_masks)
                    batch_labels = []
                    for i in range(len(batch_label_origin)):
                        artical_label = [[] for i in range(categorys)]
                        wordtokenmap, tokenwordmap = wordTokenMapping(articals[i], batch_offsets[i])
                        for samp, s_label in batch_label_origin[i]:
                            if int(samp[-1]) > len(wordtokenmap)-1:
                                break
                            start, end = wordtokenmap[int(samp[0])][0], wordtokenmap[int(samp[-1])][-1] 
                            artical_label[s_label].append([start, end])
                        for ll in artical_label:
                            if not ll:
                                ll.append((0, 0))
                        artical_label = sequence_padding(artical_label, seq_dims=1)
                        batch_labels.append(artical_label)
                        
                    batch_labels  = sequence_padding(batch_labels, seq_dims=2)
                    yield batch_token_ids, batch_masks, batch_labels
                    batch_data, batch_label_origin, articals = [], [], []
                    batch_token_ids, batch_masks, batch_offsets = [], [], []
    
token_ids = tf.keras.layers.Input(shape=(None,))
masks = tf.keras.layers.Input(shape=(None,)) 

backbone = LongformerModel.from_pretrained('allenai/longformer-base-4096', gradient_checkpointing=True)
out = backbone(token_ids, masks)[0]
out2 = EfficientGlobalPointer(categorys, 64)  #bs, label, seq_len, seq_len

def globalpointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    shape = K.shape(y_pred)
    y_true = y_true[..., 0] * K.cast(shape[2], K.floatx()) + y_true[..., 1]
    y_pred = K.reshape(y_pred, (shape[0], -1, K.prod(shape[2:])))
    loss = sparse_multilabel_categorical_crossentropy(y_true, y_pred, True)
    return K.mean(K.sum(loss, axis=1))
    
model = tf.keras.models.Model([token_ids, masks], out2)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(loss=globalpointer_crossentropy, optimizer=optimizer)

train_data = DataLoader(train_files, batch_size=8)
valid_data = DataLoader(val_files, batch_size=8)

def evaluate(y_ture, y_pred):
    data = []
    for token_ids, masks, y_true in valid_data:
        y_pred = model.predict([token_ids, masks])  # bs * label * seq_len * seq_len
        y_pred += masks * -np.inf
    shape = y_pred.shape 
    for i in range(shape[0]):
        tmp = []
        for j in range(shape[1]):
            for start, end in zip(y_pred[i][j][0], y_pred[i][j][1]):
                tmp.append((j, start, end))
        data.append(tmp)
    
        
    


                    
