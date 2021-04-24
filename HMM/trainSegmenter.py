# coding: utf-8
import numpy as np
import re
import pickle

from utils import *
from trainTagger import process_corpus


def preprocessing():
    tagged_sents = process_corpus()
    token_words = [w for sent in tagged_sents for w,t in sent]
    chinese_words = [w for w in token_words if re.match('^[\u4E00-\u9FA5]+$', w)]
    
    char_tag = []
    for word in chinese_words:
        if len(word) == 1:
            char_tag.append((word, 'S'))
        else:
            tag = ['B'] + ['M'] * (len(word) - 2) + ['E']
            char_tag += list(zip(list(word), tag))
    return char_tag


def train():
    tag_set = ['B','E','M','S']
    init_prob = np.array([-0.26268660809250016, -3.14e+100, -3.14e+100, -1.4652633398537678])

    char_tag = preprocessing()
    char_set = sorted(list(set([c for c, t in char_tag])))
    char_set.append("UNK")
    emission_prob = cal_condition_prob(tag_set, char_set, char_tag)
    
    labels = [t for c, t in char_tag]
    symbol_label = Condition(labels)
    trans_prob = cal_condition_prob(tag_set, tag_set, symbol_label, discount=0.01)
    save_parameters("data/HMMSegmenter.parameters.npz", init_prob, trans_prob, emission_prob)
    
    char_index = symbols2index(char_set)
    tag_index = index2labels(tag_set)
    with open("data/Segmenterindex.pkl", "wb") as f:
        pickle.dump(char_index, f)
        pickle.dump(tag_index, f)
    f.close()
