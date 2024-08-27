import os
import sys
import torch
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


def predict(text, model):
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
