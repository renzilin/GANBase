import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import  DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable

from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve, auc
from pathlib import Path



vocab =['Bac', 'Ent', 'Esc','Lis','Pse', 'Sal', 'Sta','Sac']

parser = argparse.ArgumentParser(description='PreTest_Predict')
parser.add_argument('--data_path', type=str, default='data/Bac_2000.npy', metavar='PATH',
                    help='your data files path (default:"data/Bac_2000.npy")')
parser.add_argument('--model_path', type=str, default='../model/Sta_model.pt', metavar='PATH',
                    help='your model files path (default: "model/Sta_model.pt")')
parser.add_argument('--out_path', type=str, default='out/', metavar='PATH',
                    help='path to save out files (default: "out/")')
parser.add_argument('--batch_size', type=int, default=200, metavar='N',
                    help='batch size (default: 200)')



class label2int:
    def __init__(self, vocab =vocab):
        self.int_map = {}
        self.base_map = {}
        for ind, base in enumerate(vocab):
            self.int_map[base] = ind
            self.base_map[ind] = base

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        ch = self.int_map[text]
        int_sequence.append(ch)
        return int_sequence
    
    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.base_map[i])
        return ''.join(string)
    
def add_first_elements(data, data_list):
    first_elements = [row[0] for row in data]
    data_list.extend(first_elements)
    
def add_chrom_elements(data, target_list):
    chrom_elements = [row[1] for row in data]
    target_list.extend(chrom_elements)

def read_file(data_file_name):
    loadData = np.load(data_file_name, allow_pickle=True)
    data_l = loadData.tolist()
    
    data_list = []
    label_list = []
    add_first_elements(data_l, data_list)
    add_chrom_elements(data_l, label_list)

    lis = []
    for line in data_list:
        l = [int(s) for s in list(line.strip().split())]
        lis.append(l)  
    
    return lis , label_list

# def data_load(res_name, target_name):
    # target_path = file_path + res_name + str(data_num) + '.npy'


def data_load(res_name, target_name, args):
    convert = label2int()
    target_path = args.data_path
    real_data_lis, label_list = read_file(target_path)
    data_label = [convert.text_to_int(i) for i in label_list]
    data_num = len(real_data_lis)
    if res_name == target_name:
        targets= [1 for _ in range(data_num)]
    else:    
        targets= [0 for _ in range(data_num)]
        
    tensor_dat= TensorDataset(torch.tensor(real_data_lis), torch.tensor(targets).long() , torch.tensor(data_label) )
    data_loader = DataLoader(dataset = tensor_dat, batch_size = args.batch_size, shuffle = False)    
    
    return data_loader


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *  -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        return self.dropout(x)

class Discriminator(nn.Module):
    def __init__(self, num_classes = 2, vocab_size = 4, embedding_dim = 4, nhead =2, dropout  = 0.2, seq_len = 200 ):
        super(Discriminator, self).__init__()
        self.hidden_dim = 64
        self.embed = nn.Embedding(vocab_size , embedding_dim)
        self.pe = PositionalEncoding(d_model = embedding_dim, dropout = dropout, max_len = seq_len)
        self.ln = nn.Linear(embedding_dim, self.hidden_dim )
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead = nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.conv1 = nn.Sequential(
                    nn.Conv1d(self.hidden_dim, 128, 5, 1, 2),  
                    nn.ReLU(),
                )
        self.conv2 = nn.Sequential(
                    nn.Conv1d(128, 256, 5, 1, 2), 
                    nn.ReLU(),
                )
        self.conv3 = nn.Sequential(
                    nn.Conv1d(256, 256, 5, 1, 2),  
                    nn.ReLU(),
                )
        self.normlayer2 = nn.LayerNorm(256)
        self.fc = nn.Linear(256 * seq_len , num_classes)
        self.activation = nn.LogSoftmax(dim=1)
    def forward(self,x):
        x = self.embed(x)  
        se = x        
        x = self.pe(x)  
        x = x + se
        x = self.ln(x)
        se = x
        x = self.transformer_encoder(x)        
        x = x + se
        x = x.permute(0,2,1)  # batch_size, hidden_dim, seq_len
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0,2,1)  # batch_size, hidden_dim, seq_len
        x = self.normlayer2(x)
        x = self.normlayer2(x)
        x =  x.contiguous().view((x.size()[0], -1)) # batch_size, seq_len*hidden_dim*2
        x = self.fc(x)
        x = self.activation(x)
        return x     

# def model_load(sname,select_epoch,args):
    # model_dir = model_path + sname + '/model_dis'+str(select_epoch)+'.pt'

def model_load(args):
    model_dir = args.model_path
    nll_loss = nn.NLLLoss()
    discriminator = Discriminator()
    dis_dict = torch.load(model_dir, map_location='cpu')# 先加载参数
    discriminator.load_state_dict(dis_dict,False)  # 再让模型加载参数, 恢复得到模型
    
    model = discriminator.cuda()
    nll_loss = nll_loss.cuda()
    cudnn.benchmark = True
    
    return model

def cal_acc(model,dataloader):
    y_true=[]
    y_pred =[]
    y_score =[]
    for data, target ,data_label in dataloader: 
        data, target, data_label= data.cuda(), target.cuda(), data_label.cuda()
        target = target.contiguous().view(-1)
        output = model(data)
        score = torch.exp(output)[:,1]
        pred = output.data.max(1)[1]
        y_true += target.cpu().tolist()
        y_pred += pred.cpu().tolist()
        y_score += score.cpu().tolist()
        
    cm=confusion_matrix(y_true, y_pred)
    # cm_dict = {'tn': cm[0, 0], 'fp': cm[0, 1],'fn': cm[1, 0], 'tp': cm[1, 1]}
    acc = accuracy_score(y_true, y_pred)
    return acc
    
    # pred, label = prediction(model, data)
    # acc = sklearn(pred, label)
    # roc_auc = roc_auc_score(y_true, y_score)
    # f1 = f1_score(y_true, y_pred)
    # precision = precision_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    # mcc=matthews_corrcoef(y_true, y_pred)
    # p, r, t = precision_recall_curve(y_true, y_score)
    # pr_auc = auc(r, p)
    # return acc,roc_auc,pr_auc,precision,recall,mcc,f1


if __name__ == '__main__':
    print('START...')

    # Parse arguments
    args = parser.parse_args()
    output = []


    target = Path(args.model_path).stem[0:3]
    rest = Path(args.data_path).stem[0:3]    

    target_model = model_load(args)

    rest_data   = data_load(rest, target, args)

    acc = cal_acc(target_model, rest_data) # sklearn

    output.append( [target, rest, acc] )


    # for target in vocab:
    #     for rest in vocab:

    #         target_model = model_load(target)

    #         rest_data   = data_load(rest, target)

    #         acc = cal_acc(target_model, rest_data) # sklearn

    #         output.append( [target, rest, acc] )
    
    df = pd.DataFrame(output, columns=['target','rest','acc'])
    print('predict_result:')
    print(df)
    out_path = args.out_path
    if os.path.exists(out_path) is not True:
      os.mkdir(out_path) 
    df.to_excel(out_path+'pre_predict-output.xlsx', index=False)  
    print('save output file.')


