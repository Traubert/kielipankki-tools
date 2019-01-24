#!/usr/bin/python3
# -*- coding: utf-8 -*-
import scipy.stats
from math import exp

dist_filenames = list(map(lambda x: '/var/www/kielipankki-tools/' + x, ["bio_ja_ymparistot_wordfreq.txt", "humanistinen_wordfreq.txt", "farmasia_wordfreq.txt", "kayttaytymistiede_wordfreq.txt", "maajametsatiede_wordfreq.txt", "matemaattis_wordfreq.txt", "oikeustiede_wordfreq.txt", "teologinen_wordfreq.txt", "valtiotiede_wordfreq.txt"]))
#  "laaketiede_wordfreq.txt",

stopwords = set(["joka",
             "on",
             "hän",
             "se",
             "että",
             "olla",
             "myös",
             "voida",
             "tämä",
             "tulla",
])

dists = []
#lowest_probs = []
allwords = set()
for filename in dist_filenames:
    dist = {}
#    lowest_prob = 1.0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                dist[parts[0]] = exp(-1*float(parts[1]))
                allwords.add(parts[0])
#                if dist[parts[0]] < lowest_prob:
#                    lowest_prob = dist[parts[0]]
    dists.append(dist)
#    lowest_probs.append(lowest_prob)

def get_divergences(lemmalist):
    global allwords
    counts = {}
    total = len(lemmalist)
    for lemma in lemmalist:
        if lemma not in stopwords:
            counts[lemma] = counts.get(lemma, 0) + 1
            allwords.add(lemma)
    minprob = 1./len(allwords)
    dist = {lemma: count/total for (lemma, count) in counts.items()}
    seq = [dist[x] if x in dist else minprob for x in allwords]
    retval = []
    for i in range(len(dist_filenames)):
        thisseq = [dists[i][x] if x in dists[i] else minprob for x in allwords]
        kl_div = scipy.stats.entropy(seq, thisseq)
        name = dist_filenames[i]
        name = name[name.rindex("/") + 1 : name.index("_")]
        retval.append((name, float(kl_div)))
    retval.sort(key = lambda x: x[1])
    return retval

if __name__ == '__main__':
    print(str(get_divergences("syöpä melanooma lääkitys löydös diskurssi substantiivi naiseus tyttöys kokemus subjekti suomi valtio vakavaraisuus tieto aavistus hokema politiikka".split())))
