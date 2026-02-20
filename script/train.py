import os
import argparse
import time
import numpy as np
from pathlib import Path
# import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset , DataLoader

from data_loader import DisDataIter, GenDataIter, EVAL_DataIter
from generator_lstm import Generator
from dis_trans_cnn import Discriminator
from target_lstm import TargetLSTM
from rollout import Rollout
from loss import PGLoss

# from tqdm import tqdm
# import sys
# CUDA_VISIBLE_DEVICES=0,1

# Arguemnts
parser = argparse.ArgumentParser(description='SeqGAN')
parser.add_argument('--project_path', type=str, default='', metavar='PATH',
                    help='Project path (default: '')')
parser.add_argument('--data_path', type=str, default="data/train_data/zymo/Saccharomyces_cerevisiae_ref_convert_step100_200bp.txt" , metavar='PATH',
                    help='data path to save files (default: '')')
parser.add_argument('--rounds', type=int, default=101, metavar='N',
                    help='rounds of adversarial training (default: 101)')
parser.add_argument('--g_pretrain_steps', type=int, default=15, metavar='N',
                    help='steps of pre-training of generators (default: 15)')
parser.add_argument('--d_pretrain_steps', type=int, default=1, metavar='N',
                    help='steps of pre-training of discriminators (default: 1)')
parser.add_argument('--update_rate', type=float, default=0.8, metavar='UR',
                    help='update rate of roll-out model (default: 0.8)')
parser.add_argument('--n_rollout', type=int, default=16, metavar='N',
                    help='number of roll-out (default: 16)')
parser.add_argument('--vocab_size', type=int, default=4, metavar='N',
                    help='vocabulary size (default: 10)')
parser.add_argument('--batch_size', type=int, default=280, metavar='N',
                    help='batch size (default: 280)')
parser.add_argument('--g_batch_size', type=int, default=50000, metavar='N',
                    help='batch size (default: 50000)')
parser.add_argument('--gen_lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate of generator optimizer (default: 1e-3)')
parser.add_argument('--dis_lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate of discriminator optimizer (default: 1e-3)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--seq_len', type=int, default=200, metavar='S',
                    help='sequence length (default: 200)')

# Genrator Parameters
g_embed_dim = 4
g_hidden_dim = 4 

# Discriminator Parameters
d_num_class = 2
d_embed_dim = 4
d_n_head = 2

def read_file(data_file):
    lis = []
    if Path(data_file).suffix == '.txt':
        with open(data_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            l = [int(s) for s in list(line.strip().split())]
            lis.append(l) 
        return lis
    if Path(data_file).suffix == '.npy':
        loadData = np.load(data_file)
        for item in loadData:
            l = item.tolist()
            lis.append(l) 
        return lis    

def train_generator_MLE(gen, data_iter, criterion, optimizer, args):    
    """
    Train generator with MLE
    """
    # print('begin--train_generator_MLE---')
    
    total_loss = 0.
    for data, target in data_iter:


        if args.cuda:
            data, target = data.to(device), target.to(device)
        target = target.contiguous().view(-1)

        output = gen(data)

        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    data_iter.reset()
    avg_loss = total_loss / len(data_iter)    #  len(data_iter) 2
    print("train loss: {:.5f}".format(avg_loss))
# gen_pretrain_train_loss.append(avg_loss)



def train_generator_PG(gen, dis, rollout, pg_loss, optimizer, args):
    """
    Train generator with the guidance of policy gradient
    """
    # construct the input to the genrator, add zeros before samples and delete the last column
    samples = gen.sample(args.batch_size, seq_len)
    print('sample',samples.size())
    zeros = torch.zeros(args.batch_size, 1, dtype=torch.int64)
    print('zeros',zeros.size()) 

    if samples.is_cuda:
        zeros = zeros.to(device)
    inputs = torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous()
    print('inputs',inputs.size())
    print('inputs',inputs)
    targets = samples.data.contiguous().view((-1,))
    print('targets',targets.size())
    print('targets',targets)

    # calculate the reward
    rewards = torch.tensor(rollout.get_reward(samples, args.n_rollout, dis))
    if args.cuda:
        rewards = rewards.to(device)
    print('reward.shape',rewards)
    # update generator
    output = gen(inputs)
    print('gen_pg_out',output.size())
    loss = pg_loss(output, targets, rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def eval_generator(model, data_iter, criterion, args):
    # print('--eval_generator--')
    """
    Evaluate generator with NLL
    """
    total_loss = 0.
    with torch.no_grad():
        for data, target in data_iter:
            if args.cuda:
                data, target = data.to(device), target.to(device)
            target = target.contiguous().view(-1)
            pred = model(data) ## ---pred--- torch.Size([6464, 4]
            loss = criterion(pred, target)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_iter)
    return avg_loss



def train_discriminator(dis, gen, criterion, optimizer,
        dis_adversarial_train_loss, dis_adversarial_train_acc, args):
    """
    Train discriminator
    """
    correct = 0
    total_loss = 0.

    for iter, real_data_batch in enumerate(real_dataloader):
        # print('iter',iter)
        mini_batch_size  = real_data_batch[0].size(0)
        # print('mini_batch_size',mini_batch_size)

        real_data_batch =  real_data_batch[0].tolist()
        gene_sample_batch = gen.sample(mini_batch_size,  seq_len).tolist()

        data_iter = DisDataIter(real_data_batch, gene_sample_batch, mini_batch_size * 2)

        i=0
        for data, target in data_iter:
            # print('i',i)
            i+=1
            if args.cuda:
                data, target = data.to(device), target.to(device)
            target = target.contiguous().view(-1)


            output = dis(data)         
            # print('output.shape',output.shape)

            pred = output.data.max(1)[1]    
            correct += pred.eq(target.data).cpu().sum()
            # print('correct',correct)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # data_iter.reset()
    
    avg_loss = total_loss / iter
    acc = correct.item() / ((iter * (args.batch_size * 2.0)+ mini_batch_size))
    print("train loss: {:.5f}, train acc: {:.3f}".format(avg_loss, acc))
    dis_adversarial_train_loss.append(avg_loss)
    dis_adversarial_train_acc.append(acc)
gene_sample_batch=[]

def eval_discriminator(model, data_iter, criterion, args):
    """
    Evaluate discriminator, dropout is enabled
    """
    correct = 0
    total_loss = 0.
    with torch.no_grad():
        for data, target in data_iter:
            if args.cuda:
                data, target = data.to(device), target.to(device)
            target = target.contiguous().view(-1)
            output = model(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            loss = criterion(output, target)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_iter)
    acc = correct.item() / data_iter.data_num
    return avg_loss, acc



def adversarial_train(gen, dis, rollout, pg_loss, nll_loss, gen_optimizer, dis_optimizer, 
        dis_adversarial_train_loss, dis_adversarial_train_acc, args):
    """
    Adversarially train generator and discriminator
    """
    # train generator for g_steps
    print("#Train generator")
    print("##G-Step")
    train_generator_PG(gen, dis, rollout, pg_loss, gen_optimizer, args)

    # train discriminator for d_steps
    print("#Train discriminator")
    print("##D-Step")
    train_discriminator(dis, gen, nll_loss, dis_optimizer, 
            dis_adversarial_train_loss, dis_adversarial_train_acc, args)

    # update roll-out model
    rollout.update_params()






if __name__ == '__main__':
    print('START...')
    # Parse arguments
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    POSITIVE_FILE = args.project_path + args.data_path
    file_name = Path(POSITIVE_FILE).stem

    pre_read_time = time.perf_counter()
    # real_data = read_file(POSITIVE_FILE)
    real_data_lis = read_file(POSITIVE_FILE)

    print('len(real_data)',len(real_data_lis))
    print('read_real_data_time',time.perf_counter() - pre_read_time)

    # real_dataset = TensorDataset(torch.tensor(real_data_lis))
    real_data = torch.tensor(real_data_lis)
    # print('real_data',real_data.shape)
    real_dataset = TensorDataset(real_data)
    real_dataloader = DataLoader(real_dataset, batch_size = args.batch_size, shuffle = True)

    log_path = args.project_path + 'logs/'
    save_path = args.project_path + 'save/'
    if os.path.exists(log_path) is not True:
        os.makedirs(log_path) 
    if os.path.exists(save_path) is not True:
        os.makedirs(save_path) 
    # Set models, criteria, optimizers
    seq_len = args.seq_len

    generator = Generator(vocab_size = args.vocab_size, embedding_dim = g_embed_dim, hidden_dim = g_hidden_dim,  use_cuda = args.cuda )
    discriminator = Discriminator(num_classes=d_num_class, vocab_size=args.vocab_size, embedding_dim = d_embed_dim, nhead = d_n_head, dropout =0.2 , seq_len = seq_len,  num_layers =6)
    target_lstm = TargetLSTM(args.vocab_size, g_embed_dim, g_hidden_dim, args.cuda)
    gen_optimizer = optim.Adam(params=generator.parameters(), lr=args.gen_lr)
    dis_optimizer = optim.SGD(params=discriminator.parameters(), lr=args.dis_lr)
    
    nll_loss = nn.NLLLoss()
    pg_loss = PGLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.cuda:
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        target_lstm = target_lstm.to(device)
        cudnn.benchmark = True


    # # 如果有多个 GPU，使用 DataParallel 进行模型并行
    # if torch.cuda.device_count() > 1:
    #     print("使用多个 GPU")
    #     generator = nn.DataParallel(generator)
    #     target_lstm = nn.DataParallel(target_lstm)
    #     discriminator = nn.DataParallel(discriminator)
    #     gen_optimizer = nn.DataParallel(gen_optimizer)
    #     dis_optimizer = nn.DataParallel(dis_optimizer)
    
    advers_writer = SummaryWriter(log_path)


    dis_adversarial_train_loss = []
    dis_adversarial_train_acc = []


    # Pre-train generator using MLE
    print('#####################################################')
    print('Start pre-training generator with MLE...')
    print('#####################################################\n')
    gen_real_data_iter = GenDataIter(real_data_lis, args.g_batch_size)
    print('real_data_dataloader_finish')
    for i in range(args.g_pretrain_steps):
        print("G-Step {}".format(i))
        pre_gen_start = time.perf_counter()
        train_generator_MLE(generator, gen_real_data_iter, nll_loss, 
            gen_optimizer, args)
        
        gen_data_samples = generator.sample(args.g_batch_size, seq_len).tolist()
        eval_iter = EVAL_DataIter(gen_data_samples)

        gen_pretrain_eval_loss= eval_generator(target_lstm, eval_iter, nll_loss, args)
        pre_gen_time = time.perf_counter()
        print('time',pre_gen_time-pre_gen_start)
        print("eval loss: {:.5f}\n".format(gen_pretrain_eval_loss))



    # Pre-train discriminator
    print('#####################################################')
    print('Start pre-training discriminator...')
    print('#####################################################\n')
    for i in range(args.d_pretrain_steps):
        print("D-Step {}".format(i))
        pre_dis_start = time.perf_counter()
        train_discriminator(discriminator, generator, nll_loss, 
            dis_optimizer, dis_adversarial_train_loss, dis_adversarial_train_acc, args)
        
        pretrain_real_data_eval = next(iter(real_dataloader))
        length = pretrain_real_data_eval[0].size(0)
        pretrain_real_data_eval = pretrain_real_data_eval[0].tolist()
        gen_data_samples = generator.sample(length, seq_len).tolist()
        eval_iter = DisDataIter(pretrain_real_data_eval, gen_data_samples, length * 2)

        dis_pretrain_eval_loss, dis_pretrain_eval_acc = eval_discriminator(discriminator, eval_iter, nll_loss, args)

        pre_dis_time = time.perf_counter()
        print('time',pre_dis_time-pre_dis_start)
        print("eval loss: {:.5f}, eval acc: {:.3f}\n".format(dis_pretrain_eval_loss, dis_pretrain_eval_acc))

    output_dis_path = args.project_path +save_path+'/model_dis_pretrain.pt'
    torch.save(discriminator.state_dict(), output_dis_path)
    output_gen_path = args.project_path +save_path+'/model_gen_pretrain.pt'
    torch.save(generator.state_dict(), output_gen_path)

 
    # Adversarial training
    print('#####################################################')
    print('Start adversarial training...')
    print('#####################################################\n')

    rollout = Rollout(generator, args.update_rate)


    for i in range(args.rounds):
        print("Round {}".format(i))
        train_start = time.perf_counter()

        adversarial_train(generator, discriminator, rollout, 
            pg_loss, nll_loss, gen_optimizer, dis_optimizer, 
            dis_adversarial_train_loss, dis_adversarial_train_acc, args) 
        
        
        train_real_data_eval = next(iter(real_dataloader))
        
        train_real_data_eval = train_real_data_eval[0].tolist()  

        train_gen_eval_data = generator.sample(args.batch_size, seq_len).tolist()
        gen_eval_iter = GenDataIter(train_gen_eval_data, args.batch_size)
        dis_eval_iter = DisDataIter(train_real_data_eval, train_gen_eval_data, args.batch_size)
        
        gen_adversarial_eval_loss = eval_generator(target_lstm, gen_eval_iter, nll_loss, args)
        
        dis_adversarial_eval_loss, dis_adversarial_eval_acc = eval_discriminator(discriminator, dis_eval_iter, nll_loss, args)

        
        train_time = time.perf_counter()
        print('time',train_time-train_start)
        print("gen eval loss: {:.5f}, dis eval loss: {:.5f}, dis eval acc: {:.3f}\n"
            .format(gen_adversarial_eval_loss, dis_adversarial_eval_loss, dis_adversarial_eval_acc))
        # 存模型
        output_dis_path = args.project_path +save_path+'/model_dis{}.pt'.format(i)
        torch.save(discriminator.state_dict(), output_dis_path)
        output_gen_path = args.project_path +save_path+'/model_gen{}.pt'.format(i) 
        torch.save(generator.state_dict(), output_gen_path)
        
        # 画折线图-----loss曲线
        advers_writer.add_scalar("Loss/dis_train", dis_adversarial_train_loss[-1], i)
        advers_writer.add_scalar("Loss/dis_eval", dis_adversarial_eval_loss, i)
        advers_writer.add_scalar("Loss/gen_eval", gen_adversarial_eval_loss, i)
        advers_writer.add_scalar("Acc/dis_train", dis_adversarial_train_acc[-1], i)
        advers_writer.add_scalar("Acc/dis_eval", dis_adversarial_eval_acc, i)

    advers_writer.close()
