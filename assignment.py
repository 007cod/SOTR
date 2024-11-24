import numpy as np
import os 
import pandas as pd
import sys
from utils import metric, Task, Worker
import copy

class KM:
    def __init__(self, n, cost_matrix):
        self.n = n  # 顶点数
        self.cost_matrix = self.adjust_cost_matrix(cost_matrix)  # 调整后的邻接矩阵（权重矩阵）
        self.match_x = [-1] * n  # X集合匹配的点
        self.match_y = [-1] * n  # Y集合匹配的点
        self.u = [0] * n  # X集合的潜力
        self.v = [0] * n  # Y集合的潜力
        self.dist = [0] * n  # 辅助数组，用于存储距离
        self.inc = sys.maxsize
        self.visited_x = [False] * self.n
        self.visited_y = [False] * self.n
    
    def adjust_cost_matrix(self, cost_matrix):
        """处理负权重：调整矩阵使其所有元素非负"""
        # 行处理：每行减去该行最小值
        min_value = np.min(cost_matrix)
        cost_matrix = cost_matrix - min_value + 0.001
        return cost_matrix
    
    def dfs(self, x):
        """寻找增广路径"""
        self.visited_x[x] = True

        for y in range(self.n):
            if self.visited_y[y]:
                continue
            gap = self.u[x] + self.v[y] - self.cost_matrix[x][y]
            if gap < 1e-4:
                self.visited_y[y] = True
                if self.match_y[y] == -1 or self.dfs(self.match_y[y]):
                    self.match_x[x] = y
                    self.match_y[y] = x
                    return True
            elif self.inc > gap:
                self.inc = gap
        return False
        

    
    def km(self):
        """KM算法主体"""
        # 先初始化u和v的值
        for x in range(self.n):
            self.u[x] = max(self.cost_matrix[x])  # 初始化u[x]为X集合中的最大边权
        
        # 进行n次尝试增广路径
        for x in range(self.n):
            self.dist = [sys.maxsize] * self.n
            while True:
                self.inc = sys.maxsize
                self.visited_x, self.visited_y = [False] * self.n, [False] * self.n

                if self.dfs(x):
                    break
                for j in range(self.n):
                    if self.visited_x[j]:
                        self.u[j] -= self.inc
                    if self.visited_y[j]:
                        self.v[j] += self.inc
        
        # 计算最大匹配的成本
        total_cost = 0
        for x in range(self.n):
            if self.match_x[x] != -1:
                total_cost += self.cost_matrix[x][self.match_x[x]]
        
        return total_cost, self.match_x
    


class SRTS():
    def __init__(self, tasks:list[Task], workers:list[Worker], pref_matrix, emission, dis, args, **kwarg) -> None:
        '''
        tasks: [time, x, y, deadline, payoff], workers: [time, x, y, range, capacity, deadline]
        '''
        self.tasks = tasks
        self.workers = workers
        self.pref_matrix = pref_matrix
        self.emission = emission
        self.dis = dis
        self.args = args
        
        self.k = self.args['k']
        self.RS = [[] for _ in range(len(self.workers))]
        self.RTW = [[] for _ in range(len(self.tasks))]
        self.SumSR = np.zeros((len(self.tasks)))
        
        self.profit = 0
        self.cost = 0
        self.__dict__.update(kwarg)
        
    def get_RS(self, ):
        for w_i, w in enumerate(self.workers): 
            for s_j, s in enumerate(self.tasks):
                dis = self.dis[w_i][s_j]
                # arrive_time = dis/self.args["v"] + w.publish_time
                arrive_time = self.time_cost[w_i][s_j] + w.publish_time
                
                if dis < self.args["range"] and  arrive_time < w.deadline + w.publish_time and arrive_time < s.deadline + w.publish_time and arrive_time > s.publish_time and self.emission[w_i, s_j] < 1e9:
                    if self.emission[w_i, s_j] < 0:
                        print("eee")
                    self.RS[w_i].append(s_j)
    
    # def distance(self, a, b):
    #     return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2) + 1e-8
        
    def gain(self, s_id, w_id, idx): 
        if idx < self.args["k"]:
            last = self.temp_s[s_id] / (1 - self.pref_matrix[w_id][s_id])
            return self.temp_s[s_id] - last
        else:
            new = self.temp_s[s_id] * (1 - self.pref_matrix[w_id][s_id])
            return new - self.temp_s[s_id]

    def max_profit(self, ):
        self.temp_s = np.ones((len(self.tasks)))
        for w_id, w in enumerate(self.workers):
            for s_id in self.RS[w_id][:self.args["k"]]:
                self.temp_s[s_id] *= 1 - self.pref_matrix[w_id][s_id]
        #get max profit
        iter_num = 0
        flag = True
        while(iter_num < self.args["max_iter_num"] and flag):
            iter_num += 1
            flag = False
            for w_id, w in enumerate(self.workers):
                temp_gain = []
                for idx, s_id in enumerate(self.RS[w_id]):
                    temp_gain.append(self.gain(s_id, w_id, idx))
                new_RS = [x for _, x in sorted(zip(temp_gain, self.RS[w_id]))]
                for idx, s_id in enumerate(new_RS):
                    if idx < self.args["k"] and s_id not in self.RS[w_id][:self.args["k"]]:
                        self.temp_s[s_id] *= 1 - self.pref_matrix[w_id][s_id]
                        flag = True
                    elif idx >= self.args["k"] and s_id in self.RS[w_id][:self.args["k"]]:
                        self.temp_s[s_id] /= 1 - self.pref_matrix[w_id][s_id]
                        flag = True
                self.RS[w_id] = new_RS
                
    def min_cost(self, KMcost):
        self.recommend_task_seqs = [x[:self.args["k"]] for x in self.RS]
        
        self.temp_cost = np.zeros((len(self.tasks)))
        self.SumSR = np.zeros((len(self.tasks)))
        for w_id, w in enumerate(self.workers):
            for rank, s_id in enumerate(self.recommend_task_seqs[w_id]):
                self.SumSR[s_id] += (self.args["k"] - rank)
                self.temp_cost[s_id] += (self.args["k"] - rank) * self.emission[w_id][s_id] * (1 - self.temp_s[s_id])
                
        for s_id in range(len(self.temp_cost)):
            if self.SumSR[s_id] == 0:
                self.temp_cost[s_id] = 0
            else:
                self.temp_cost[s_id] /= self.SumSR[s_id]
        
        def cost_up(w_id, rank, new_rank):
            s_id = self.recommend_task_seqs[w_id][rank]
            if rank==new_rank:
                return 0
            if self.SumSR[s_id] + (rank - new_rank) == 0:
                print("11")
            return (self.temp_cost[s_id] * self.SumSR[s_id] + (rank - new_rank) * self.emission[w_id][s_id] * (1 - self.temp_s[s_id]))/(self.SumSR[s_id] + (rank - new_rank)) - self.temp_cost[s_id]
                    #get min cost
        iter_num = 0
        flag = True
        while(iter_num < self.args["max_iter_num"] and flag):
            iter_num += 1
            flag = False
            for w_id, w in enumerate(self.workers):
                re_length = len(self.recommend_task_seqs[w_id])
                if re_length == 0:
                    continue
                if KMcost:
                    mp = np.zeros((re_length, re_length))
                    for idx in range(re_length):
                        for j in range(re_length):
                            mp[idx][j] = cost_up(w_id, idx, j)
                            
                    km = KM(re_length, -mp)
                    total_cost, match_result = km.km()
                    last_rec = copy.deepcopy(self.recommend_task_seqs[w_id])
                    
                    #更新变量
                    for i in range(re_length):
                        s_id = self.recommend_task_seqs[w_id][i]
                        new_id  = match_result[i]
                        self.SumSR[s_id] += i - new_id
                        self.temp_cost[s_id] += mp[i][new_id]
                    for i in range(re_length):
                        self.recommend_task_seqs[w_id][match_result[i]] = last_rec[i]
                    if last_rec != self.recommend_task_seqs[w_id]:
                        flag = True
                else:
                    for idx in range(re_length-1, 0,  -1):
                        for j in range(0, idx-1):
                            add1 = cost_up(w_id, j, j+1)
                            add2 = cost_up(w_id, j+1, j)
                            if add1 + add2 < 0:
                                #更新变量
                                self.SumSR[self.recommend_task_seqs[w_id][j]] -= 1
                                self.SumSR[self.recommend_task_seqs[w_id][j+1]] += 1
                                self.temp_cost[self.recommend_task_seqs[w_id][j]] += add1
                                self.temp_cost[self.recommend_task_seqs[w_id][j+1]] += add2
                                self.recommend_task_seqs[w_id][j], self.recommend_task_seqs[w_id][j+1] = self.recommend_task_seqs[w_id][j+1], self.recommend_task_seqs[w_id][j]
                                flag = True    
    
    def assign(self, name):
        if name == "STRS":
            return self.STR(KMcost=False)
        elif name == "STR":
            return self.STR(KMcost=True)
        elif name == "MTR":
            return self.assign_no_cost()
        elif name == "TopK":
            return self.topK()
        elif name == "GCA":
            return self.GCA()
                    
    def STR(self, KMcost=False):
        self.get_RS()
        self.max_profit()
        
        self.min_cost(KMcost)
        
        self.profit = sum([1 - x for x in self.temp_s])
        self.cost = sum(self.temp_cost)
        
        print(f"profit:{self.profit}, cost:{self.cost}")
        return self.recommend_task_seqs
    
    def assign_no_cost(self, ):
        self.get_RS()
        self.max_profit()
        self.recommend_task_seqs = [sorted(x[:self.args["k"]], reverse=True) for x in self.RS]
        return self.recommend_task_seqs
    
    def topK(self, ):
        self.get_RS()
        for w_id, task_set in enumerate(self.RS):
            self.RS[w_id] = [x for _, x in sorted(zip(self.pref_matrix[w_id, task_set], task_set), reverse=True)]
        self.recommend_task_seqs = [x[:self.args["k"]] for x in self.RS]
        
        return self.recommend_task_seqs
    
    def GCA(self, ):
        self.get_RS()
        self.recommend_task_seqs = [[] for _ in range(len(self.workers))]
        for w_i, w in enumerate(self.workers):
            for s_id in self.RS[w_i]:
                self.RTW[s_id].append(w_i)
        
        for s_id, w_set in enumerate(self.RTW):
            self.RTW[s_id] = [x for _, x in sorted(zip(self.pref_matrix[w_set, s_id], w_set), reverse=True)]
        
        lenR = [len(x) for x in self.RTW]
        
        num = 0
        while num < len(self.workers) * self.k:
            s_id = np.argmin(lenR)
            if lenR[s_id] == 1e9:
                break
            flag = 0
            # if s_id == 1004:
            #     print('fef')
            for id, w_i in enumerate(self.RTW[s_id]):
                if len(self.recommend_task_seqs[w_i]) < self.k:
                    self.recommend_task_seqs[w_i].append(s_id)
                    lenR[s_id] -=1
                    self.RTW[s_id].pop(id)
                    flag = 1
                    num +=1
                    break
            if flag == 0:
                lenR[s_id] = 1e9
                self.RTW[s_id] = []
        return self.recommend_task_seqs