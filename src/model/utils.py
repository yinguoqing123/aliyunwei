import torch
from torch.nn import functional as F
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split

def  macro_f1(overall_df, pred):
    """
    计算得分
    :param target_df: [sn,fault_time,label]
    :param submit_df: [sn,fault_time,label]
    :return:
    """

    weights =  [5/11,  4/11,  1/11,  1/11]

    macro_F1 =  0.
    for i in  range(len(weights)):
        TP =  len(overall_df[(overall_df['label'] == i) & (overall_df[pred] == i)])
        FP =  len(overall_df[(overall_df['label'] != i) & (overall_df[pred] == i)])	
        FN =  len(overall_df[(overall_df['label'] == i) & (overall_df[pred] != i)])
        precision = TP /  (TP + FP)  if  (TP + FP)  >  0  else  0
        recall = TP /  (TP + FN)  if  (TP + FN)  >  0  else  0
        F1 =  2  * precision * recall /  (precision + recall)  if  (precision + recall)  >  0  else  0
        macro_F1 += weights[i]  * F1
        print(f"class {i}, precision: {precision}, recall: {recall}, F1: {F1}")
    return macro_F1  

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.5, emb_name=['emb1', 'emb2', 'emb3', 'emb4', 'emb5']):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'emb' in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name=['emb1', 'emb2']):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'emb' in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {} 
    