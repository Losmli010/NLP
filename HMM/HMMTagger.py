# coding: utf-8

from utils import *
import pickle
from viterbi import viterbi

def tagger(sentence):
    Pi,A,B=load_parameters("data/HMMTagger.parameters.npz")
    with open("data/Taggerindex.pkl","rb") as f:
        word_index=pickle.load(f)
        label_index=pickle.load(f)
    f.close()
    obs=map_obs(word_index,sentence)
    prob,route=viterbi(obs,Pi,A,B)
    sequence=[label_index[i] for i in route]
    result=''
    for word,tag in zip(sentence,sequence):
        result+=''.join(' '+word+'/'+tag+' ')
    return result.strip()