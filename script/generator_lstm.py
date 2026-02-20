import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """ Generator 
        3 layers 
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, use_cuda):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = 3, batch_first=True)

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.init_params()

    def forward(self, x):

        self.lstm.flatten_parameters()
        h0, c0 = self.init_hidden(x.size(0))

        emb = self.embed(x) 
        out, _ = self.lstm(emb,(h0, c0)) 
        out = self.log_softmax(self.fc(out.contiguous().view(-1, self.hidden_dim))) 

        return out

    def step(self, x, h, c):
       
        self.lstm.flatten_parameters()
        emb = self.embed(x) 
        out, (h, c) = self.lstm(emb, (h, c)) 
        out = self.log_softmax(self.fc(out.contiguous().view(-1, self.hidden_dim))) 
        return out, h, c

    def init_hidden(self, batch_size):
        h = torch.zeros(3, batch_size, self.hidden_dim)
        c = torch.zeros(3, batch_size, self.hidden_dim)
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c
    
    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def sample(self, batch_size, seq_len, x=None):

        samples = []
        if x is None:
            h, c = self.init_hidden(batch_size)
            x = torch.zeros(batch_size, 1, dtype=torch.int64)
            if self.use_cuda:
                x = x.cuda()
            for _ in range(seq_len):
                out, h, c = self.step(x, h, c)
                prob = torch.exp(out)
                x = torch.multinomial(prob, 1)
                samples.append(x)
        else:
            h, c = self.init_hidden(x.size(0))
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            for i in range(given_len):
                out, h, c = self.step(lis[i], h, c)
                samples.append(lis[i])
            prob = torch.exp(out)
            x = torch.multinomial(prob, 1)
            for _ in range(given_len, seq_len):
                samples.append(x)
                out, h, c = self.step(x, h, c)
                prob = torch.exp(out)
                x = torch.multinomial(prob, 1)
        out = torch.cat(samples, dim=1) # along the batch_size dimension
        return out
