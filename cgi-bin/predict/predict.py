#!/usr/bin/env python
import torch
import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import hfst

class VecReader:
    def __init__(self, indexfile = "s24_surface_vecs.hofst",
                 vecfile = "binary_vecs.bin",
                 dim = 128):
        self.index = hfst.HfstTransducer.read_from_file(indexfile)
        self.vecs = open(vecfile, "rb")
        self.dimension = dim
        self.cache = {}

    def find(self, word):
        if word in self.cache:
            return self.cache[word]
        idx = self.index.lookup(word)
        if len(idx) == 0:
            vec = np.zeros(self.dimension)
            print("word " + word + " not found!")
            self.cache[word] = vec
            return vec
        self.vecs.seek(2 * self.dimension * int(idx[0][0]))
        vec = np.fromfile(self.vecs,
                          dtype=np.float16,
                          count=self.dimension)
        self.cache[word] = vec
        return vec


class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        self.vecreader = word2vec.VecReader()
        
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

#        self.embed = nn.Embedding(V, D)
#        self.embed = nn.Embedding.from_pretrained(torch.zeros(V, D))

#        self.embed = lambda x: (torch.zeros(len(x), max(map(len, x)), D))
#        self.embed = nn.Embedding.from_pretrained()
#        self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def embed(self, x):
        examplevecs = []
        for example in x:
            examplevecs.append([])
            for word in example.split():
                examplevecs[-1].append(self.vecreader.find(word))
        def pad(l, to, padding):
            while len(l) < to:
                l.append(padding)
        maxlen = max(map(len, x))
        padding = [0.0 for i in range(len(examplevecs[0][0]))]
        for example in examplevecs:
            pad(example, maxlen, padding)
        return torch.Tensor(examplevecs)
        
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = list(x)
            x = self.embed(x)  # (N, W, D)
        
        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit


class Args:
    def __init__(self):
        self.lr = 0.001
        self.epochs = 256
        self.batch_size = 64
        self.log_interval = 1
        self.test_interval = 100
        self.save_interval = 500
        self.save_dir = 'snapshot'
        self.early_stop = 1000
        self.save_best = True
        self.shuffle = False
        self.dropout = 0.5
        self.max_norm = 3.0
        self.embed_dim = 128
        self.kernel_num = 100
        self.kernel_sizes = [2,3,4,5]
        self.static = False
        self.device = -1
        self.no_cuda = False
        self.snapshot = '/var/www/cgi-bin/predict/snapshot/2018-08-16_06-35-21/best_steps_2700.pt'
        self.predict = None
        self.test = False
        self.class_num = 3

import torch.autograd as autograd
import torch.nn.functional as F


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        batch_num = 0
        while batch_num * args.batch_size < len(train_iter):
            batch = train_iter[batch_num * args.batch_size : (batch_num + 1) * args.batch_size]
            batch_num += 1
            feature, target = list(map(lambda x: x[0], batch)), torch.LongTensor(list(map(lambda x: int(x[1])-1, batch)))
#            feature.data.t_(), target.data.sub_(1)  # batch first, index align

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/len(batch)
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                         loss.data[0], 
                                                                         accuracy,
                                                                         corrects,
                                                                         len(batch)))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            if steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(data, model, args):
    model.eval()
    feature, target = list(map(lambda x: x[0], data)), torch.LongTensor(list(map(lambda x: int(x[1])-1, data)))
    logit = model(feature)
    loss = F.cross_entropy(logit, target, size_average=False)
    avg_loss = loss.data[0]
    corrects = (torch.max(logit, 1)
                [1].view(target.size()).data == target.data).sum()
    size = len(data)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy


def train_predict(text, model):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
#    text = text_field.preprocess(text)
#    text = [[text_field.vocab.stoi[x] for x in text]]
#    x = text_field.tensor_type(text)
    x = model.embed([text])
    x = autograd.Variable(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return str(predicted.item() + 1)


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


def predict(texts):
    args = Args()
    try:
        cnn = CNN_Text(args)
    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        sys.stdout.buffer.write(str(ex).encode("utf-8"))
    cnn.load_state_dict(torch.load(args.snapshot))
    resultdict = {'1': 'negative', '2': 'neutral', '3': 'positive'}
    return list(map(lambda x: resultdict[train_predict(x, cnn)], texts))
