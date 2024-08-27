import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from . import embutils
import numpy as np

torch.set_num_threads(8)

class CNN_Text(nn.Module):

    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        self.embs = embutils.WordEmbeddings()
        self.embs.load_from_file("s24_smiley/s24_surfaces_normalized.bin")
        self.embedding_mismatches = set()
        
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
            for word in example:
                word_and_vec = self.embs.get_embedding(word)
                if word != word_and_vec[0]:
                    self.embedding_mismatches.add((word, word_and_vec[0]))
                examplevecs[-1].append(np.array(word_and_vec[1]))
        maxlen = max(map(len, examplevecs))
        def pad(l, to, padding):
            while len(l) < to:
                l.append(padding)
        padding = np.zeros(self.args.embed_dim)
        for example in examplevecs:
            pad(example, max(5, maxlen), padding)
        return torch.Tensor(examplevecs)
        
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = list(x)
        x = self.embed(x)  # (N, W, D)
        
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
