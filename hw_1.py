# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:06:19 2022

@author: 80463
"""
import numpy as np
import scipy.io as scio

class HMM:
    def __init__(self, A, B, pi):
        # 观测符号集合表示 Vk = {A,B,C,D,E,F} --> Vk = {0,1,2,3,4,5}
        # 状态转移矩阵
        self.A = np.array(A, np.float64)
        # 观测概率矩阵
        self.B = np.array(B, np.float64)
        # 初始概率分布
        self.pi = np.array(pi, np.float64)

        # 状态的数量，A:N*N，shape[0]返回矩阵A的行数
        self.N = self.A.shape[0]
        # 观测符号的数量，A:N*M，shape[0]返回矩阵A的列数
        self.M = self.B.shape[1]
        
    # 输出HMM的参数信息
    def printHMM(self):
        print ("==================================================")
        print ("HMM content: N =",self.N,",M =",self.M)
        for i in range(self.N):
          if i==0:
            print ("hmm.A ",self.A[i,:]," hmm.B ",self.B[i,:])
          else:
            print ("   ",self.A[i,:],"    ",self.B[i,:])
        print ("hmm.pi",self.pi)
        print ("==================================================")

    # 前向算法
    # obs_seq：观察值序列
    def Forward(self,obs_seq):
        T = len(obs_seq)
        alpha = np.zeros((T,self.N), np.float64)
        # 初值
        for i in range(self.N):
            alpha[0,i] = self.pi[i] * self.B[i,obs_seq[0]]
        # 递归
        for t in range(T-1):
            for j in range(self.N):
                sum = 0.0
                for i in range(self.N):
                    sum += alpha[t,i] * self.A[i,j]
                alpha[t+1,j] = sum * self.B[j,obs_seq[t+1]]
        # 终止
        '''
        sum = 0.0
        for i in range(self.N):
            sum += alpha[T-1,i]
        prob = sum 
        '''
        return alpha
    
    # 后向算法
    def Backward(self,obs_seq):
        T = len(obs_seq)
        beta = np.zeros((T,self.N), np.float64)
        for i in range(self.N):
            beta[T-1,i] = 1.0
        for t in range(T-2,-1,-1):
            for i in range(self.N):
                sum = 0.0
                for j in range(self.N):
                    sum += beta[t+1,j]*self.A[i,j] * self.B[j,obs_seq[t+1]]
                beta[t,i] = sum
        '''
        sum = 0.0
        for i in range(self.N):
            sum += beta[0,i]
        prob = sum
        '''
        return beta
    
    # Viterbi算法
    def viterbi(self,obs_seq):
        T = len(obs_seq)
        # 初值
        delta = np.zeros((T,self.N),np.float64) # t时刻状态i，最可能从t-1时刻状态j转移而来，记录最大的概率值
        psi = np.zeros((T,self.N),np.float64)   # t时刻状态i，最可能从t-1时刻状态j转移而来，记录最大转移的状态序号
        q = np.zeros(T)
        for i in range(self.N):
            delta[0,i] = self.pi[i] * self.B[i,obs_seq[0]]
            psi[0,i] = 0
        # 递推
        for t in range(T):
            for j in range(self.N):
                for i in range(self.N):
                    delta[t,j] = self.B[j,obs_seq[t]] * np.array(delta[t-1,i] * self.A[i,j], np.float64).max()
                    psi[t,j] = np.array(delta[t-1,i] * self.A[i,j]).argmax()
        # 终止,最后时刻的最高得分和最佳状态
        pstar = delta[T-1,:].max()
        qstar = delta[T-1,:].argmax()
        # 最优路径回溯
        for t in range(T-2,-1,-1):
            q[t] = psi[t+1,q[t+1]]
        return pstar, qstar, q

    # Baum-Welch算法
    def BaumWelch(self,observations):
        n_obs =  len(observations)
        T = len(observations[0,:])
        for n in range(n_obs):
            obs_seq = observations[n,:]
            alpha = self.Forward(obs_seq)
            beta = self.Backward(obs_seq)

            # calculate Gamma
            gamma = np.zeros((T,self.N), np.float64)
            for t in range(T):
                sum = 0.0
                for i in range(self.N):
                    gamma[t,i] = alpha[t,i] * beta[t,i]
                    sum += gamma[t,i]
                for j in range(self.N):
                    gamma[t,j] = gamma[t,j] / sum
                    
            # calculate ksi
            ksi = np.zeros((T-1,self.N,self.N), np.float64)

            for t in range(T-1):
                sum = 0.0
                for i in range(self.N):
                    for j in range(self.N):
                        ksi[t,i,j] = alpha[t,i] * self.A[i,j] * self.B[j,obs_seq[t+1]] * beta[t+1,j]
                        sum += ksi[t,i,j]
                for i in range(self.N):
                    for j in range(self.N):
                        ksi[t,i,j] = ksi[t,i,j] / sum
                        
            # update model
            denominatorA = np.zeros((self.N),np.float64)       # 更新后aij分母
            denominatorB = np.zeros((self.N),np.float64)       # 更新后bjk分母
            numeratorA = np.zeros((self.N,self.N),np.float64)  # 更新后aij分子
            numeratorB = np.zeros((self.N,self.M),np.float64)  # 更新后bjk分子
            newpi = np.zeros((self.N),np.float64)
            newA = np.zeros((self.N,self.N),np.float64)
            newB = np.zeros((self.N,self.M),np.float64)
            # ! 均值更好？？
            for i in range(self.N):
                newpi[i] = gamma[0,i]
                for t in range(T):
                    denominatorA[i] += gamma[t,i]
                    denominatorB[i] += gamma[t,i]
                # denominatorB[i] += gamma[T-1,i]

                for j in range(self.N):
                    for t in range(T-1):
                        numeratorA[i,j] += ksi[t,i,j]

                for k in range(self.M):
                    for t in range(T):
                        if obs_seq[t] == k:
                              numeratorB[i,k] += gamma[t,i]
            # print("Stop")
        for i in range(self.N):
                for j in range(self.N):
                    newA[i,j] = numeratorA[i,j] / denominatorA[i]
                for k in range(self.M):
                    newB[i,k] = numeratorB[i,k] / denominatorB[i]
            
        self.A,self.B,self.pi = newA,newB,newpi
 
            
if __name__ == "__main__":
    # A = [
    #     [0.3,0.3,0.1,0.1	,0.1,0.1],
    #     [0.1,0.3,0.3,0.1	,0.1,0.1],
    #     [0.1,0.1,0.3,0.3	,0.1,0.1],
    #     [0.1,0.1,0.1,0.3	,0.3,0.1],
    #     [0.1,0.1,0.1,0.1	,0.3,0.3],
    #     [0.3,0.1,0.1,0.1	,0.1,0.3],
    # ]
    # B = [
    #     [0.2,0.2,0.1,0.1	,0.1,0.1],
    #     [0.2,0.2,0.2,0.2	,0.1,0.1],
    #     [0.2,0.2,0.2,0.2	,0.2,0.2],
    #     [0.2,0.2,0.2,0.2	,0.2,0.2],
    #     [0.1,0.1,0.2,0.2	,0.2,0.2],
    #     [0.1,0.1,0.1,0.1	,0.2,0.2],
    # ]
    A = np.zeros((6,6))
    B = np.zeros((6,6))
    pi = [0.2,0.1,0.2,0.2,0.2,0.1]
    hmm = HMM(A,B,pi)
    
    data_path = 'model.mat'
    data = scio.loadmat(data_path)
    O = data.get('model_2')
    
    hmm.BaumWelch(O)
    hmm.printHMM()
