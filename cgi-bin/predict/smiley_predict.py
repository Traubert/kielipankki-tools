#!/usr/bin/env python3
import argparse
import torch

from s24_smiley import model
from s24_smiley import train
from s24_smiley import data

parser = argparse.ArgumentParser(description='CNN text classifier')

parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 10]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-early-stop', type=int, default=10000, help='iteration numbers to stop without performance increasing [default: 10000]')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
args = parser.parse_args()

# update args and print
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

args.class_num = 3
# model
cnn = model.CNN_Text(args)
cnn.load_state_dict(torch.load("s24_smiley/best_steps_2100.pt"))

def predict(texts):
    labels = ["negative", "neutral", "positive"]
    cnn.eval()
    retval = []
    for text in texts:
        score_tensor, max_index_tensor = train.predict(text, cnn)
        retval.append(labels[max_index_tensor.item()])
    return retval

def predict_batch(texts):
    labels = ["negative", "neutral", "positive"]
    return list(map(lambda x: labels[x], train.predict_batch(texts, cnn)))

if __name__ == "__main__":
    print(predict(["hyvä hyvä hyvä".split()]))
