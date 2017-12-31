# coding: utf-8

import numpy as np
import math
from collections import defaultdict

def ConditionalFreqDist(labels_set,symbols_set,corpus):
    """
    type:list(tuple(str,str))
    rtype:numpy array
    """
    count=defaultdict(int)
    for (symbol,label) in corpus:
        count[(symbol,label)]+=1
        
    N=len(labels_set)
    M=len(symbols_set)
    arr=np.zeros((N,M),np.int)
    for i,symbol in enumerate(symbols_set):
        for j,label in enumerate(labels_set):
            arr[j][i]=count[(symbol,label)]                 
    return arr

def AbsoluteDiscounting(arr,discount=0.75):
    """
    绝对折扣平滑法
     对所有统计次数大于0的项减去-discount
    """
    if arr.size==len(arr):
        smooth_arr=np.array([0.0]*len(arr),np.float)
        N_zero=list(arr).count(0)
        for i in range(len(arr)):
            if arr[i]==0:
                smooth_arr[i]=(len(arr)-N_zero)*discount/N_zero
            else:
                smooth_arr[i]=arr[i]-discount
    else:
        (row,column)=arr.shape
        smooth_arr=np.zeros((row,column),np.float)
        for i in range(row):
            N_zero=list(arr[i]).count(0)
            for j in range(column):
                if arr[i][j]==0:
                    smooth_arr[i][j]=(column-N_zero)*discount/N_zero
                else:
                    smooth_arr[i][j]=arr[i][j]-discount
    return smooth_arr

def logprob(prob):
    return math.log(prob,math.e)

def ConditionalProbDist(arr):
    """
    type:numpy array,frequency distribution
    rtype:numpy array,probability distribution
    """
    (row,column)=arr.shape
    prob_arr=np.zeros((row,column),np.float)
    for i in range(row):
        N=sum(arr[i])
        for j in range(column):
            prob_arr[i][j]=logprob(arr[i][j]/N)
    return prob_arr

def cal_condition_prob(labels_set,symbols_set,symbol_label,discount=0.75):
    return ConditionalProbDist(AbsoluteDiscounting(ConditionalFreqDist(labels_set,symbols_set,symbol_label),discount))

def Condition(labels):
    """
    type:list(label)
    rtype:bigram,list(tuple(str,str))
    """
    condition_label=[]
    i=0
    while i<len(labels)-1:
        condition_label.append((labels[i+1],labels[i]))
        i+=1
    return condition_label
    
def FreqDist(symbols_set,symbols):
    """
    type:list(symbol)
    rtype:1-dim array
    """
    count=defaultdict(int)
    for symbol in symbols:
        count[symbol]+=1
    
    freq_arr=np.array([0.0]*len(symbols_set),np.int)
    for i,symbol in enumerate(symbols_set):
        freq_arr[i]=count[symbol]
    return freq_arr

def ProbDist(freq_arr):
    """
    type:1-dim array,frequency distribution
    rtype:1-dim array,probability distribution
    """
    prob_arr=np.array([0.0]*len(freq_arr),np.float)
    N=sum(freq_arr)
    for i in range(len(freq_arr)):
        prob_arr[i]=logprob(freq_arr[i]/N)
    return prob_arr

def save_parameters(outfile,init_prob,trans_prob,emission_prob):
    np.savez(outfile,
             Pi=init_prob,
             A=trans_prob,
             B=emission_prob)
    print("Saved model parameters to %s." % outfile)   

def load_parameters(path):
    npzfile=np.load(path)
    init_prob,trans_prob,emission_prob=npzfile["Pi"],npzfile["A"],npzfile["B"]
    return init_prob,trans_prob,emission_prob

def symbols2index(symbols_set):
    """
    建立字典索引
    rtype:list
    """
    symbol_index={}
    for i,symbol in enumerate(symbols_set):
        symbol_index[symbol]=i
    return symbol_index

def index2labels(labels_set):
    label_index={}
    for i,label in enumerate(labels_set):
        label_index[i]=label
    return label_index

def map_obs(symbol_index,obs):
    obs2index=[]
    for s in obs:
        if s in symbol_index:
            obs2index.append(symbol_index[s])
        else:
            obs2index.append(symbol_index["UNK"])
    return obs2index