import argparse
import math

import torch
import torch.nn as nn

from torch.autograd import Variable


parser = argparse.ArgumentParser(description='Prediction')
parser.add_argument('--data', type=str, default='ATTGTACTTCGTTCAATCACTTCCGGTATTTGTACTTCGTTCAGTTTTCAAATGAAGGTAGGTGTTTAACCTCGATTCCGTTTGTAGTCGTCTGGTTTTTTCGCATTTATCGTGAAACGCTTTCGCGTTTTTCGTGCGCCGCTTCATTAGTTATATTATTAAATATTAACTAATGTGTGCTCTATATTTATTGAATAGTT', metavar='data',
                    help='input sigle sequence data')
parser.add_argument('--model_path', type=str, default='model/Human_model.pt', metavar='PATH',
                    help='your model files paths (default: "model/Human_model.pt")')
parser.add_argument('--length', type=int, default=200, metavar='N',
                    help='length (default: 200)')
parser.add_argument('--option', type=str, default='depletion', 
                    help='choose depletion or enrichment target species (default: "depletion")')

class label2int:
    def __init__(self, baseseq = 'ACGT'):
        self.int_map = {}
        self.base_map = {}
        for ind, base in enumerate(baseseq):
            self.int_map[base] = ind
            self.base_map[ind] = base
    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            ch = self.int_map[c]
            int_sequence.append(ch)
        return int_sequence

def data_load(args):
    length = args.length
    convert = label2int()
    predict_data = convert.text_to_int(args.data[:length])
    return torch.tensor([predict_data])

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
        self.normlayer = nn.LayerNorm(256)
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
        x = x.permute(0,2,1) 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0,2,1) 
        x = self.normlayer(x)
        x = self.normlayer(x)
        x =  x.contiguous().view((x.size()[0], -1)) 
        x = self.fc(x)
        x = self.activation(x)
        return x

def model_load(args):
    model_dir = args.model_path
    discriminator = Discriminator().cuda()
    dis_dict = torch.load(model_dir)
    discriminator.load_state_dict(dis_dict,False)  
    return discriminator

def pred_result(model,data):
    data = data.cuda()
    output = model(data)
    pred = output.data.max(1)[1]
    return pred
    

if __name__ == '__main__':
    print('START...')
    args = parser.parse_args()
    
    target_model = model_load(args)
    input_data   = data_load(args)
    result = pred_result(target_model, input_data) 

    if args.option == 'depletion':
        print('result:',(1-result.item()))
    else :
        print('result:',result.item())

