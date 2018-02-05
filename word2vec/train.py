# coding: utf-8

import os
import logging
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

class sentence_iter(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), "r", encoding="utf-8"):
                yield line.split()
                
def train_word2vec(filename):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    sentence = LineSentence(filename)
    model = Word2Vec(sentence, size=256, window=5, min_count=5, workers=4)
    model.save("data/zhwiki.word2vec.model")
    model.wv.save_word2vec_format("data/zhwiki_word2vec.txt", binary=False)