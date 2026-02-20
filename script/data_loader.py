import math
import random
import torch
from torch.utils.data import Dataset, DataLoader


class EVAL_DataIter:
    """ Toy data iter to load digits """ ### 

    def __init__(self, gen_data):
        super(EVAL_DataIter, self).__init__()
        self.data_lis = gen_data
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.idx = 0
        self.reset()

    def __len__(self):
        return self.data_num

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.data_num]
        d = [self.data_lis[i] for i in index]
        d = torch.tensor(d)

        # 0 is prepended to d as start symbol
        data = torch.cat([torch.zeros(len(index), 1, dtype=torch.int64), d], dim=1)

        # print('data_iter__target',data.size())
        target = torch.cat([d, torch.zeros(len(index), 1, dtype=torch.int64)], dim=1)
        # print('data_iter__target',target.size())
        
        self.idx += self.data_num
        return data, target


"""
 ###############################
     GenDataIter--MLE--train
 ###############################
"""

class GenDataIter:
    """ Toy data iter to load digits """ ### 

    def __init__(self, data, batch_size):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        # self.data_lis = self.read_file(data_file)
        self.data_lis = data
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.num_batches = math.ceil(self.data_num / self.batch_size)
        self.idx = 0
        self.reset()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.batch_size]
        # print('index',index)
        d = [self.data_lis[i] for i in index]
        d = torch.tensor(d)
        # print('d',d.shape)
        # print('d[1:]',d[:,1:].shape)
        # print('0s',(torch.zeros(len(index), 1, dtype=torch.int64)).shape)

        # 0 is prepended to d as start symbol
        data = torch.cat([torch.zeros(len(index), 1, dtype=torch.int64), d[:,1:]], dim=1)

        # print('data_iter__target',data.size())
        target = d
        # print('data_iter__target',target.size())
        
        self.idx += self.batch_size
        return data, target



#############################
#    MLE_train_real_data    #
#############################



# class GenDataIter:
#     """ Toy data iter to load digits """ ### 

#     def __init__(self, data):
#         super(GenDataIter, self).__init__()
#         # self.batch_size = batch_size
#         # # self.data_lis = self.read_file(data_file)
#         self.data_lis = data
#         self.data_num = len(self.data_lis)
#         self.indices = range(self.data_num)
#         self.num_batches = math.ceil(self.data_num / self.batch_size)
#         self.idx = 0
#         self.reset()

#     def __len__(self):
#         return self.num_batches

#     def __iter__(self):
#         return self

#     def __next__(self):
#         return self.next()
    
#     def reset(self):
#         self.idx = 0
#         random.shuffle(self.data_lis)

#     def next(self):
#         if self.idx >= self.data_num:
#             raise StopIteration
#         index = self.indices[self.idx : self.idx + self.batch_size]
#         d = [self.data_lis[i] for i in index]
#         d = torch.tensor(d)

#         # 0 is prepended to d as start symbol
#         data = torch.cat([torch.zeros(len(index), 1, dtype=torch.int64), d], dim=1)

#         # print('data_iter__target',data.size())
#         target = torch.cat([d, torch.zeros(len(index), 1, dtype=torch.int64)], dim=1)
#         # print('data_iter__target',target.size())
        
#         self.idx += self.batch_size
#         return data, target
    




"""
 #########################
     DisDataIter
 #########################
"""


class DisDataIter:
    """ Toy data iter to load digits """

    def __init__(self, real_data, fake_data, batch_size):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        real_data_lis = real_data
        fake_data_lis = fake_data
        # print(len(real_data_lis),len(fake_data_lis))
        self.data = real_data_lis + fake_data_lis
        self.labels = [1 for _ in range(len(real_data_lis))] +\
                        [0 for _ in range(len(fake_data_lis))]
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs)
        self.indices = range(self.data_num)
        self.num_batches = math.ceil(self.data_num / self.batch_size)
        self.idx = 0
        self.reset()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        data = torch.tensor(data)
        label = torch.tensor(label)
        self.idx += self.batch_size
        return data, label











# class DisDataIter:
#     """ Toy data iter to load digits """

#     def __init__(self, real_data, fake_data, batch_size):
#         super(DisDataIter, self).__init__()
#         self.batch_size = batch_size
#         real_data_lis = real_data
#         fake_data_lis = fake_data
#         print(len(real_data_lis),len(fake_data_lis))
#         self.data = real_data_lis + fake_data_lis
#         self.labels = [1 for _ in range(len(real_data_lis))] +\
#                         [0 for _ in range(len(fake_data_lis))]
#         self.pairs = list(zip(self.data, self.labels))
#         self.data_num = len(self.pairs)
#         self.indices = range(self.data_num)
#         self.num_batches = math.ceil(self.data_num / self.batch_size)
#         self.idx = 0
#         self.reset()

#     def __len__(self):
#         return self.num_batches

#     def __iter__(self):
#         return self

#     def __next__(self):
#         return self.next()
    
#     def reset(self):
#         self.idx = 0
#         random.shuffle(self.pairs)

#     def next(self):
#         if self.idx >= self.data_num:
#             raise StopIteration
#         index = self.indices[self.idx : self.idx + self.batch_size]
#         pairs = [self.pairs[i] for i in index]
#         data = [p[0] for p in pairs]
#         label = [p[1] for p in pairs]
#         data = torch.tensor(data)
#         label = torch.tensor(label)
#         self.idx += self.batch_size
#         return data, label



"""
 #########################
     MINIBATCH_DisDataIter
 #########################
"""

class Minibatch_DisDataIter:
    """ Toy data iter to load digits """

    def __init__(self, real_data, fake_data):
        super(Minibatch_DisDataIter, self).__init__()
        real_data_lis = real_data
        fake_data_lis = fake_data
        # print(len(real_data_lis),len(fake_data_lis))
        # print(real_data_lis[0], fake_data_lis[0])
        self.data = real_data_lis + fake_data_lis
        self.labels = [1 for _ in range(len(real_data_lis))] +\
                        [0 for _ in range(len(fake_data_lis))]
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs)
        self.indices = range(self.data_num)
        self.idx = 0
        self.reset()

    def __len__(self):
        return self.data_num

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.data_num]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        data = torch.tensor(data)
        label = torch.tensor(label)
        self.idx += self.data_num
        return data, label





#####

#####



class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]