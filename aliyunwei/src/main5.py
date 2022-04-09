#  bert 再来最后一次
import torch
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
import numpy as np
from torchtext.vocab import vocab
from torch.utils.data import Dataset, DataLoader
import math 
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import json
from sklearn.model_selection import train_test_split, KFold
from ema import EMA
from torch.optim.lr_scheduler import *
from utils import MyDataSet4, macro_f1, DiceLoss

# bert encode
class BertLayer(nn.Module):
    def __init__(self, hidden_size=40, heads=4, intermediate_size=20):
        super().__init__()
        self.attention = BertAttention(heads, hidden_size)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(intermediate_size, hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        #outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output,) 
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertAttention(nn.Module):
    def __init__(self, heads, hidden_size):
        super().__init__()
        self.self = BertSelfAttention(heads, hidden_size)
        self.output = BertSelfOutput(hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size=32):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.ln(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states
    
class BertOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        #self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        #hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states + input_tensor

class BertSelfAttention(nn.Module):
    def __init__(self, heads=3, head_size=48):
        assert head_size % heads == 0, 'the size is not compatable'
        super().__init__()
        self.num_attention_heads = heads
        self.attention_head_size = head_size // heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(head_size, self.all_head_size)
        self.key = nn.Linear(head_size, self.all_head_size)
        self.value = nn.Linear(head_size, self.all_head_size)
        self.ln = nn.LayerNorm(head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        hidden_states = self.ln(hidden_states)
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs
    
class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        #self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        #hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states + input_tensor


train_set = pd.read_csv('../data/train_set6.csv')
submit_df = pd.read_csv('../data/submit_df6.csv')

lookup1 = torch.load('../data/word2idx6.pk')
intervalbucketnum, cntbucketnum, durationbucketnum = 101, 23, 12

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.emb1 = nn.Embedding(len(lookup1), 10, padding_idx=0)
        self.emb2 = nn.Embedding(88, 40, padding_idx=0)
        self.emb3 = nn.Embedding(intervalbucketnum, 5)
        self.emb4 = nn.Embedding(cntbucketnum, 3)
        self.emb5 = nn.Embedding(durationbucketnum, 2)
        self.att = BertLayer(40, 4, 40)
        self.att2 = BertLayer(40, 4, 40)
        #self.position_emb = nn.Embedding(51, 64)
        #self.att = BertAttention(1, 64)
        #self.dense1 = nn.Linear(64, 32)
        #self.dense2 = nn.Linear(32, 4)
        #self.att2 = BertAttention(1, 64)
        
        #self.position_emb_init(64)
        #self.register_buffer("position_ids", torch.arange(51).expand((1, -1)))
        self.dense2 = nn.Linear(40, 4)
        self.dense2.bias.data = torch.tensor([-2.38883658, -1.57741002, -0.57731536, -1.96360971])

        
    def forward(self, feat):
        feat, server_model, len_seq, mask = feat  # len1 batch_size * sentence_num
        feat1, feat2, feat3, feat4 = feat[..., :3], feat[..., 3], feat[..., 4], feat[..., 5]
        word_emb = self.emb1(feat1)  # (b, s, 3, d)
        b, s, w, d = word_emb.shape
        server_model = self.emb2(server_model)
        emd_interval = self.emb3(feat2)
        emb_cnt = self.emb4(feat3)
        emb_duration = self.emb5(feat4)
        #server_model2 = self.dense1(server_model)
        word_emb = word_emb.reshape(b, s, w*d)
        word_emb = torch.concat([word_emb, emd_interval, emb_cnt, emb_duration], dim=-1)
        #position_emb = self.position_emb(self.position_ids[:, :s+1])
        # word_emb attention
        att_emb = torch.concat([server_model.unsqueeze(dim=-2), word_emb], dim=-2)
        #att_emb = att_emb + position_emb
        att_mask = torch.concat([torch.ones(b, 1), mask], dim=-1)
        att_mask = self.get_extended_attention_mask(att_mask, (b, s))
        att_emb = self.att(att_emb, att_mask)[0]
        att_emb = self.att2(att_emb, att_mask)[0][:, 0, :]
        #score = torch.relu(self.dense1(att_emb))
        #score = self.dense2(score)
        score = self.dense2(att_emb)
        return score
    
    def orthogonal_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                torch.nn.init.orthogonal_(param)
            if 'bias' in name:
                torch.nn.init.zeros_(param)
                
    def position_emb_init(self, dim):
        self.position_emb.weight.requires_grad = False
        self.position_emb.weight.data = self.positionalencoding1d(dim, 51)
                
    def manual_embeddings(self, featmatrix1, featmatrix2):
        self.emb3 = nn.Embedding(len(lookup1), 4, padding_idx=0)
        self.emb4 = nn.Embedding(len(lookup1), 4, padding_idx=0)
        self.emb3.weight.requires_grad = False
        self.emb3.weight.data = featmatrix1
        self.emb4.weight.requires_grad = False
        self.emb4.weight.data = featmatrix2
    
    def positionalencoding1d(self, d_model, length):
        """
            :param d_model: dimension of the model
            :param length: length of positions
            :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe
    
    def get_extended_attention_mask(self, attention_mask, input_shape) :
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

def lr_lambda(epoch):
    if epoch < 5:
        return 0.05
    else:
        return 0.4

model = Model()
def train_and_evaluate(train_set_, test_set_, submit_set_, name):
    model = Model() 
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=4)
    #lr_sche = LambdaLR(optimizer, lr_lambda)
    scheduler = ExponentialLR(optimizer, 0.9)
    scheduler1 = LambdaLR(optimizer, lr_lambda)
    train_set_ = MyDataSet4(train_set_)
    test_df = test_set_
    test_set_ = MyDataSet4(test_set_, mode='test')
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 2.0, 1.0, 1.0]))
    #criterion = DiceLoss(weight=torch.tensor([1.5, 1.5, 1.0, 1.0]), coff1=1, coff2=0.5)
    train_data = iter(train_set_)
    test_data = iter(test_set_)
    best_f1 = 0
    for epoch in range(30):
        running_loss = 0
        for step in range(train_set_.step_max):
            feat, label = next(train_data)
            model.train()
            pred = model(feat)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % 50 == 49:
                print(f"Epoch {epoch+1}, step {step+1}: {running_loss}")
                running_loss = 0 
        scheduler1.step()   #warm up 
        if epoch >= 5:
            scheduler.step() 
        # epoch 结束后 evaluate
        preds = []
        with torch.no_grad():
            for step in range(test_set_.step_max):
                model.eval()
                feat, label = next(test_data)
                pred = model(feat).argmax(dim=-1).numpy()
                preds.extend(pred)
        test_df['pred'] = preds
        macro_F1 =  macro_f1(test_df)
        if macro_F1 > best_f1:
            best_f1 = macro_F1
            torch.save(model.state_dict(), f'../model5/model_{name}.pt')
            test_df.to_csv(f"../valid5/pred_{name}.csv", index=False)
        print(f"macro F1: {macro_F1}")
        #scheduler.step(macro_F1)
        print(f"max macor F1: {best_f1}")
    submit_set = MyDataSet4(submit_set_, mode='predict')
    submit_set_iter = iter(submit_set)
    preds = []
    model.load_state_dict(torch.load(f'../model5/model_{name}.pt'))
    with torch.no_grad():
        model.eval()
        for step in range(submit_set.step_max):
            feat = next(submit_set_iter)
            pred = model(feat)
            pred = torch.softmax(pred, dim=-1).numpy()
            pred = [json.dumps(p.tolist()) for p  in pred]
            preds.extend(pred)
    submit_set_[f'label_{name}'] = preds
    submit_set_.to_csv(f'../submit5/submit_{name}.csv', index=False)
    
for i, (train_idx, test_idx) in enumerate(KFold(shuffle=True, random_state=2022).split(train_set[['sn', 'fault_time','feature', 'servertype', 'label']])):
    train_set_ = train_set.iloc[train_idx]
    train_set_ = pd.concat([train_set_, train_set_[train_set_.label==0]]).reset_index(drop=True)
    test_set_ = train_set.iloc[test_idx]     
    train_and_evaluate(train_set_, test_set_, submit_df, i)
    print('=====================================')