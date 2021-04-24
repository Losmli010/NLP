# coding: utf-8
import numpy as np


class BaumWelch(object):
    def __init__(self, N, M, obs):
        self.N = N                               # 状态数
        self.M = M                               # 观测信号数
        self.obs = obs                             # 观测数据
        self.T = len(obs)                         # 观测序列长度
        
    def forward(self):
        """
        前向算法
        """
        self.alpha = np.zeros((self.T, self.N), np.float)

        # 计算初始值
        for i in range(self.N):
            self.alpha[0][i] = self.Pi[i] * self.B[i][self.obs[0]]

        # 递推计算
        for t in range(1, self.T):
            for i in range(self.N):
                sum_val = 0
                for j in range(self.N):
                    sum_val += self.alpha[t - 1][j] * self.A[j][i]
                self.alpha[t][i] = sum_val * self.B[i][self.obs[t]]
    
    def backward(self):
        """
        后向算法
        """
        self.beta = np.zeros((self.T, self.N), np.float)

        # 初始化
        for i in range(self.N):
            self.beta[self.T - 1][i] = 1
 
        # 递推
        for t in range(self.T - 2, -1, -1):
            for i in range(self.N):
                for j in range(self.N):
                    self.beta[t][i] += self.A[i][j] * self.B[j][self.obs[t + 1]] * self.beta[t + 1][j]
                
    # 初始化λ =（A, B, Pi）
    def init(self):
        """
        随机生成 A，B，Pi
        并保证每列相加等于 1
        """    
        self.A = np.zeros((self.N, self.N), np.float)        # 状态转移概率矩阵
        self.B = np.zeros((self.N, self.M), np.float)        # 观测概率矩阵
        self.Pi = np.array([0.0] * self.N, np.float)           # 初始状态概率矩阵
        
        for i in range(self.N):
            random_list = np.random.randint(0, 100, size=self.N)
            for j in range(self.N):
                self.A[i][j] = random_list[j] / sum(random_list)

        for i in range(self.N):
            random_list = np.random.randint(0, 100, size=self.M)
            for j in range(self.M):
                self.B[i][j] = random_list[j] / sum(random_list)

        random_list = np.random.randint(0, 100, size=self.N)
        for i in range(self.N):
            self.Pi[i] = random_list[i] / sum(random_list)
        print(self.A, self.B, self.Pi)

    def ksi(self, t, i, j):
        """
        计算ksi
        """
        numerator = self.alpha[t][i] * self.A[i][j] * self.B[j][self.obs[t + 1]] * self.beta[t + 1][j]
        denominator = 0
    
        for i in range(self.N):
            for j in range(self.N):
                denominator += self.alpha[t][i] * self.A[i][j] * self.B[j][self.obs[t + 1]] * self.beta[t + 1][j]
   
        return numerator / denominator

    def gamma(self, t, i):
        """
        计算γ
        """
        numerator = self.alpha[t][i] * self.beta[t][i]
        denominator = 0
    
        for j in range(self.N):
            denominator += self.alpha[t][j] * self.beta[t][j]
        
        return numerator / denominator
    
    def em(self, Maxsteps=100):
        self.init()
        step = 0
        
        while step < Maxsteps:
            step += 1
            print(step)
            
            temp_A = np.zeros((self.N, self.N), np.float)
            temp_B = np.zeros((self.N, self.M), np.float)
            temp_Pi = np.array([0.0] * self.N, np.float)
            
            self.forward()
            self.backward()
            
            # a(ij)
            for i in range(self.N):
                for j in range(self.N):
                    numerator = 0.0
                    denominator = 0.0
                    for t in range(self.T - 1):
                        numerator += self.ksi(t, i, j)
                        denominator += self.gamma(t, i)
                    temp_A[i][j] = numerator / denominator
                    
            # b(ij)
            for j in range(self.N):
                for k in range(self.M):
                    numerator = 0.0
                    denominator = 0.0
                    for t in range(self.T):
                        if k == self.obs[t]:
                            numerator += self.gamma(t, j)
                        denominator += self.gamma(t, j)
                    temp_B[j][k] = numerator / denominator
                    
            # π(i)
            for i in range(self.N):
                temp_Pi[i] = self.gamma(0, i)

            self.A = temp_A
            self.B = temp_B
            self.Pi = temp_Pi
            print(self.A,self.B, self.Pi)
