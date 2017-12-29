# coding: utf-8

import numpy as np

def viterbi(obs,Pi,A,B):
    """
    维特比算法
    """
    N=len(A)
    T=len(obs)
    V=np.zeros((T,N),np.float)
    path=np.zeros((T,N),np.int)
    
    #初始化
    for i in range(N):
        V[0][i]=Pi[i]*B[i][obs[0]]
        path[0][i]=0
        
    #递推
    for t in range(1,T):
        for i in range(N):
            max_val=0
            state=-1
            for j in range(N):
                if V[t-1][j]*A[j][i]>max_val:
                    max_val=V[t-1][j]*A[j][i]
                    state=j
            V[t][i]=max_val*B[i][obs[t]]
            path[t][i]=state
          
    #最优路径的概率  
    prob=0
    state=-1
    for i in range(N):
        if V[T-1][i]>prob:
            prob=V[T-1][i]
            state=i
    
    #逆向求最优路径
    route=[None]*T
    i = T - 1
    while i >= 0:
        route[i] = state
        state = path[i][state]
        i -= 1
    return (prob, route)