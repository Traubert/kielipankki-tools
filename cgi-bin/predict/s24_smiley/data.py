import re
import os
import random
import sqlite3

def normalize(l):
    retval = []
    for i in range(len(l)):
        w = l[i]
        if (i == 0 or (len(l[i-1]) > 0 and l[i-1][-1] in '.!?>":')) and w.istitle():
            retval.append(w.lower())
        else:
            retval.append(w)
    return retval

def preprocess(tokenized_text):
    return normalize(tokenized_text)
    
class s24_smiley():

    def __init__(self, path = None):
        self.dirname = './'
        path = self.dirname if path is None else path
        conn = sqlite3.connect(path + 's24_smileys.db')
        db_cursor = conn.cursor()
        db_cursor.execute("select count(*) from messages where class = 'positive';")
        positive_set_count = db_cursor.fetchone()[0]
        db_cursor.execute("select count(*) from messages where class = 'negative';")
        negative_set_count = db_cursor.fetchone()[0]
        db_cursor.execute("select count(*) from messages where class = 'neutral';")
        neutral_set_count = db_cursor.fetchone()[0]
        num_examples = min(positive_set_count, negative_set_count, neutral_set_count)
        self.examples = []
        self.classes = [0, 1, 2]
        for sample in db_cursor.execute(
                '''SELECT tokenized_content FROM messages where class = 'positive' ORDER BY random() LIMIT ?;''', (num_examples,)):
            self.examples.append((preprocess(sample[0]), 2))
        for sample in db_cursor.execute(
                '''SELECT tokenized_content FROM messages where class = 'negative' ORDER BY random() LIMIT ?;''', (num_examples,)):
            self.examples.append((preprocess(sample[0]), 0))
        for sample in db_cursor.execute(
                '''SELECT tokenized_content FROM messages where class = 'neutral' ORDER BY random() LIMIT ?;''', (num_examples,)):
            self.examples.append((preprocess(sample[0]), 1))
        print("Read " + str(len(self.examples)) + " examples")
                
    def split(self, dev_ratio=.975):
        random.shuffle(self.examples)
        dev_index = max(int(dev_ratio*len(self.examples)), len(self.examples) - 750)
        return self.examples[:dev_index], self.examples[dev_index:]
