import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
 
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes, vocab_size, embedding_dim, nhead, dropout , seq_len):
        super(Discriminator, self).__init__()
        self.hidden_dim = 64

        self.embed = nn.Embedding(vocab_size , embedding_dim)


        ######## positioal Embedding ########
        self.pe = PositionalEncoding(d_model = embedding_dim, dropout = dropout, max_len = seq_len)

        self.ln = nn.Linear(embedding_dim, self.hidden_dim )

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead = nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        ######## layer CNN ########

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
 
        ######## layer norm ########

        self.normlayer = nn.LayerNorm(256)

        ######## fc ########

        self.fc = nn.Linear(256 * seq_len , num_classes)

        self.activation = nn.LogSoftmax(dim=1)

    def forward(self,x):
        ######## Embedding ########
        x = self.embed(x)  

        ######## Pe ########
        se = x        
        x = self.pe(x)  
        x = x + se
        
        x = self.ln(x)
        se = x
        x = self.transformer_encoder(x)        
        x = x + se

        ######## conv  ########
        x = x.permute(0,2,1)  
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0,2,1) 

        ####### layer norm ########
        x = self.normlayer(x)

        ######## fc ########
        x =  x.contiguous().view((x.size()[0], -1)) # batch_size, seq_len*hidden_dim*2
        x = self.fc(x)

        x = self.activation(x)
        return x     


