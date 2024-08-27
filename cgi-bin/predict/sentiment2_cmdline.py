#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
import sentiment
#import smiley_predict

batch_size = 128

def get_predictions(texts):
    return sentiment.predict_batch(texts) #zip(smiley_predict.predict_batch(texts), sentiment.predict_batch(texts))

writefile = open(sys.argv[1] + "_annotated.txt", "w", encoding = "utf-8")

i = 0
this_batch = []
this_sentence = []
this_tokens = []

for line in open(sys.argv[1], 'r', encoding='utf-8'):
    if i % 3 == 0:
        if not line.startswith('/appl/kielipankki/Suomi24/2017H2/'):
            print(i)
            print(line)
            exit()
        if len(this_sentence) > 0:
            this_batch.append(this_sentence)
            if len(this_batch) >= batch_size:
                sentences_and_predictions = zip(this_batch, get_predictions(this_tokens))
                for s_p in sentences_and_predictions:
                    writefile.write(s_p[0][0])
                    writefile.write(s_p[0][1])
                    writefile.write(s_p[1] + '\n')# + ' ' + s_p[1] + '\n')
                this_batch = []
                this_tokens = []
        this_sentence = [line]
    elif i % 3 == 1:
        this_sentence.append(line)
    else:
        this_tokens.append(line.strip().split(' '))
    i += 1
this_batch.append(this_sentence)
if len(this_batch) > 0:
    sentences_and_predictions = zip(this_batch, get_predictions(this_tokens))
    for s_p in sentences_and_predictions:
        writefile.write(s_p[0][0])
        writefile.write(s_p[0][1])
        writefile.write(s_p[1] + '\n')
