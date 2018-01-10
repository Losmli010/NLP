# coding: utf-8

from utils import *
from viterbi import viterbi
import pickle
import re

def tokenizer(sentence):
    re_chinese = re.compile(r"([\u4E00-\u9FA5]+)")
    re_english = re.compile(r"([0-9A-Za-z\.']+)")
    block = re.split(re_chinese,sentence)
    tokens = []
    for s in block:
        if re.match(re_chinese,s):
            tokens += [s[i] for i in range(len(s))]
        else:
            tem = re.split(re_english,s)
            tokens += [x for x in tem if x]
    return tokens

def cut(sentence):
    Pi,A,B=load_parameters("data/HMMSegmenter.parameters.npz")
    with open("data/Segmenterindex.pkl","rb") as f:
        char_index = pickle.load(f)
        label_index = pickle.load(f)
    f.close()
    tokens = tokenizer(sentence)
    obs = map_obs(char_index,tokens)
    prob,route = viterbi(obs,Pi,A,B)
    sequence = [label_index[i] for i in route]
    seg = ''
    for char,tag in zip(tokens,sequence):
        if tag=='B':
            seg += ''.join(' '+char)
        elif tag=='M':
            seg += ''.join(char)
        elif tag=='E':
            seg += ''.join(char+' ')
        else:
            seg += ''.join(' '+char+' ')
    return seg.strip()
