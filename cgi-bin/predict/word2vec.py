# -*- coding: utf-8 -*-
import sys
import numpy as np
import hfst

class VecReader:
    def __init__(self, indexfile = "s24_surface_vecs.hofst",
                 vecfile = "binary_vecs.bin",
                 dim = 128):
        self.index = hfst.HfstTransducer.read_from_file(indexfile)
        self.vecs = open(vecfile, "rb")
        self.dimension = dim
#        self.cache = {}

    def find(self, word):
#        if word in self.cache:
#            return self.cache[word]
        idx = self.index.lookup(word)
        if len(idx) == 0:
            vec = np.zeros(self.dimension)
            print("word " + word + " not found!")
#            self.cache[word] = vec
            return vec
        self.vecs.seek(2 * self.dimension * int(idx[0][0]))
        vec = np.fromfile(self.vecs,
                          dtype=np.float16,
                          count=self.dimension)
#        self.cache[word] = vec
        return vec

#vr = VecReader()
