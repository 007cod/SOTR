import numpy as np
import os 
import pandas as pd
import math

def metric(recommend_task_seqs, pref_matrix, transport_cost, RS, k):
    """
    retrun  profit, cost
    """
    profit = 0
    cost = 0
    task_recommended_workers = {}
    DCG = [0 for _ in range(len(recommend_task_seqs))]
    SR = {}
    for w_id, rts in enumerate(recommend_task_seqs):
        best_dcg = 0
        for id, pre in enumerate(sorted(pref_matrix[w_id, RS[w_id]], reverse=True)):
            if id >= k:
                break
            best_dcg += (pow(2, pre) - 1 )/(1 + math.log2(id + 1))
        for rank, s in enumerate(rts):
            DCG[w_id] += (pow(2, pref_matrix[w_id][s]) - 1 )/(1 + math.log2(rank + 1))
            if task_recommended_workers.get(s):
                task_recommended_workers[s] += [(w_id, rank),]
                SR[s] += k - rank
            else:
                task_recommended_workers[s] = [(w_id, rank),]
                SR[s] = k - rank
        if best_dcg == 0:
            DCG[w_id] = 1
        else:
            DCG[w_id] = DCG[w_id]/best_dcg
    
    for s_id, rtw in task_recommended_workers.items():
        temp_profit = 1
        temp_cost = 0
        for w_id, rank in rtw:
            temp_profit *=  1 - pref_matrix[w_id][s_id]
            temp_cost += (k - rank)*transport_cost[w_id][s_id]/SR[s_id]
            
        profit += 1 - temp_profit
        cost += temp_cost * (1 - temp_profit)

    return profit, cost, np.mean(DCG)




def topKmetric(pred, lable, k):
    num = 0
    top_k_indices = np.argsort(pred, axis=1)[:, -k:]
    for i in range(len(lable)):
        if lable[i] in top_k_indices[i]:
            num += 1
    return num/len(lable)


class Task():
    def __init__(self, publish_time, x, y, deadline, poi, **kwarg) -> None:
        self.publish_time = publish_time
        self.x = x
        self.y = y
        self.deadline = deadline
        self.poi = poi
        self.__dict__.update(kwarg)
            

class Worker():
    def __init__(self, start_time, x, y, deadline, poi, his_poi_list, range, **kwarg) -> None:
        self.publish_time = start_time
        self.x = x
        self.y = y
        self.deadline = deadline
        self.poi = poi  #当前所在poi下标
        self.his_poi_list = his_poi_list #历史轨迹
        self.range = range

        self.__dict__.update(kwarg)