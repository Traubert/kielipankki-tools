import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import random
from . import data

torch.set_num_threads(8)

first = lambda x: x[0]
second = lambda x: x[1]

def shrinkto(x, a1, a2, b1, b2):
    fromrange = abs(a2 - a1)
    xpos = float(x - a1) / fromrange
    torange = abs(b1 - b2)
    return b1 + xpos * torange

def train(train_data, dev_data, model, args):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        print("\nStarting epoch " + str(epoch))
        batch_num = 0
        random.shuffle(train_data)
        while batch_num * args.batch_size < len(train_data):
            batch = train_data[batch_num * args.batch_size : (batch_num + 1) * args.batch_size]
            batch_num += 1
            feature = list(map(first, batch))
            target = torch.LongTensor(list(map(second, batch)))
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)
                            [1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/len(batch)
                sys.stdout.write(
                    '\rBatch {}, training step {} - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(batch_num,
                                                                                              steps,
                                                                                              loss.data.item(), 
                                                                                              accuracy,
                                                                                              corrects,
                                                                                              len(batch)))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_data, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
                        return
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)

def eval(data, model, args):
    model.eval()
    feature = list(map(first, data))
    target = torch.LongTensor(list(map(second, data)))
    logit = model(feature)
    loss = F.cross_entropy(logit, target)
    avg_loss = loss.data.item()
    corrects = (torch.max(logit, 1)
                [1].view(target.size()).data == target.data).sum()
    size = len(feature)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


def predict(text, model):
#    feature = [text.split()]
    logit = model([data.preprocess(text)])
#    print(logit)
    return torch.max(logit, 1)

def predict_batch(texts, model):
    model.eval()
#    feature = [text.split()]
    logit = model(texts)
    
#    logit = autograd.variable(logit)
#    output = model(logit)
    _, predicted = torch.max(logit, 1)
    return predicted
#    print(logit)
#    return torch.max(logit, 1)

def score_for_class_n(text, model, n):
    model.eval()
    tokens = data.preprocess(text.strip())
    while len(tokens) < 5:
        tokens.append('.')
    logit = model([tokens])
    return(logit[0][n].item())

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
