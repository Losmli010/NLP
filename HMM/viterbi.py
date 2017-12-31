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
        V[0][i]=Pi[i]+B[i][obs[0]]
        path[0][i]=0
        
    #递推
    for t in range(1,T):
        for i in range(N):
            val =[V[t-1][j]+A[j][i] for j in range(N)]
            V[t][i] =max(val)+B[i][obs[t]]
            path[t][i] =np.argmax(val)
          
    #最优路径的概率  
    last =[V[T-1][i] for i in range(N)]
    prob =max(last)
    state =np.argmax(last)
    
    #逆向求最优路径
    route=[None]*T
    i = T - 1
    while i >= 0:
        route[i] = state
        state = path[i][state]
        i -= 1
    return (prob, route)