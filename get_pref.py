import numpy as np
import pandas as pd
import torch
import heapq
from collections import defaultdict
from SOTR.util import metric, StandardScaler, DataLoaderM, load_pickle, load_adj

def normalize_pref_matrix(pref_matrix):
    # 计算每行的最大值
    row_max = np.max(pref_matrix, axis=-1, keepdims=True)  # 保持原有的形状

    # 处理最大值为0的情况，避免除以0
    row_max[row_max == 0] = 1

    # 每一行除以该行的最大值
    # pref_matrix = pref_matrix / row_max
    
    # pref_matrix[pref_matrix == 1] = pref_matrix[pref_matrix == 1] - 1e-5
    
    print(np.max(pref_matrix))
    print(np.sum(pref_matrix > 0))
    
    mean = np.mean(pref_matrix[pref_matrix > 0])
    
    pref_matrix = pref_matrix/mean
    
    pref_matrix = pref_matrix + 0.3
    pref_matrix[pref_matrix >= 1] = 1-1e-5
    print(mean)
    
    return pref_matrix

def pref(workers, tasks, graph_adj, graph_weight):
    n = len(workers)
    m = len(tasks)
    pref_matrix = np.zeros((n,m))
    graph_adj = graph_adj.transpose()
    graph = {}
    for i, e in enumerate(graph_adj):
            # pref_matrix[e[0]][e[1]] = graph_weight[i]
            graph[(e[0], e[1])]= graph_weight[i]
    for w_id, w in enumerate(workers):
        for t_id, t in enumerate(tasks):
            if graph.get((w.poi, t.poi)):
                # print(graph[(w.poi, t.poi)])
                pref_matrix[w_id][t_id] = graph[(w.poi, t.poi)]
                
    pref_matrix = normalize_pref_matrix(pref_matrix)
    
    return pref_matrix

def get_trans_model(workers, tasks, num_trans=3):
    from collections import Counter
    transport_model = np.random.randint(low=0, high=2, size=(len(workers), len(tasks)))
    # for w_id, w in enumerate(workers):
    #     transp_list = w.transp_list
    #     # 使用Counter统计每个元素出现的次数
    #     counter = Counter(transp_list)

    #     # 找到出现次数最多的元素
    #     most_common_element, count = counter.most_common(1)[0]
    #     for t_id, t in enumerate(tasks):
    #         transport_model[w_id][t_id] = most_common_element
    
    return transport_model

def emission_cost(dis, v, vehicle_p):
    v = v*3600
    emission = 0
    for p in vehicle_p:
        ef = (p[0]*v**2 + p[1]*v + p[2] + p[3]/v)*(1 - p[7])/(p[4]*v**2+p[5]*v+p[6])
        emission += ef * dis
    return emission

def dijkstra(p, poi_map, speed_pred, speed_real, start):
    n = len(poi_map)
    # 初始化最短距离数组
    dist = np.full((n, 4), 1e9)  # [预测碳排放, 时间, 真实碳排放, 距离]
    dist[start] = [0, 0, 0, 0]  # 起始节点到自身距离为0

    st = np.zeros((n))
    # 使用优先队列
    priority_queue = [(0, 0, 0, 0, start)]  # (预测碳排放, 时间, 真实碳排放, 距离, 节点)

    while priority_queue:
        curr_dist, curr_time, real_dis, dis, u = heapq.heappop(priority_queue)

        # 如果已经便利过，则跳过
        if st[u]:
            continue
        else:
            st[u] = 1

        # 更新与该节点相邻的未访问节点的最短距离
        for v in range(n):
            if poi_map[u][v] < 1e9: 
                time_id = int(min(curr_time // 300, 11))
                cost = emission_cost(poi_map[u][v], speed_pred[u][v][time_id], p)
                cost_real = emission_cost(poi_map[u][v], speed_real[u][v][time_id], p)
                if cost < 0:
                    print("erro")

                if cost < 1e9 and cost > 0:
                    new_cost = dist[u][0] + cost
                    new_time = dist[u][1] + poi_map[u][v] / speed_pred[u][v][time_id]
                    new_real_cost = dist[u][2] + cost_real
                    new_dis = dist[u][3] + poi_map[u][v]

                    # 如果找到更短的路径，更新距离
                    if new_cost < dist[v][0]:
                        dist[v] = [new_cost, new_time, new_real_cost, new_dis]
                        if st[v]==0:
                            heapq.heappush(priority_queue, (new_cost, new_time ,new_real_cost, new_dis, v))

    return dist

def get_emission(workers, tasks, poi_map, speed_pred, speed_real, time_list, pred_speed):
    # t, n, sl = speed.shape
    # for w_id, w in enumerate(workers):
    #     dist[w.poi] = dijkstra(edges, w.poi, n)
    #     for t_id, t in enumerate(tasks):
    #         dis = dist[w.poi][t.poi]
    #         emission[w_id][t_id] =  dis* transport_cost[transport_model[w_id][t_id] - 1 ]
    
    emission = np.full((len(workers), len(tasks)), np.inf)
    time_cost = np.full((len(workers), len(tasks)), np.inf)
    emission_real = np.full((len(workers), len(tasks)), np.inf)
    dis = np.full((len(workers), len(tasks)), np.inf)
    
    
    for w_id, w in enumerate(workers):
        time_id = int((w.publish_time - time_list[0])//300 - 11)
        time_id = min(len(speed_pred)-1, time_id)
        temp_speed_pred = (speed_pred[time_id, :, np.newaxis, :] + speed_pred[time_id, np.newaxis, :, :]) / 2
        
        temp_speed_real = (speed_real[time_id, :, np.newaxis, :] + speed_real[time_id, np.newaxis, :, :]) / 2
        
        dist_emission = dijkstra(w.p, poi_map, temp_speed_pred, temp_speed_real, w.poi)
        
        task_poi_list = [s.poi for s in tasks]
        dis[w_id, :] = poi_map[w.poi, task_poi_list]
        
        emission[w_id, :] =  dist_emission[task_poi_list, 0]
        time_cost[w_id, :] =  dist_emission[task_poi_list, 1]
        emission_real[w_id, :] =  dist_emission[task_poi_list, 2]
        dis[w_id, :] = dist_emission[task_poi_list, 3]
    return emission, time_cost, emission_real, dis  # 返回所有点对之间的最短路径矩阵
 
def pred_speed(model, dataloader, scaler, pred_speed, device):

    dataloader["x"][..., 0] = scaler.transform(dataloader['x'][..., 0])

    dataloader['loader'] = DataLoaderM(dataloader['x'], dataloader['y'], 64)
    dataloader['scaler'] = scaler
    
    outputs = []
    realy = torch.Tensor(dataloader['y']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]
    realy = realy / 3600
    
    if pred_speed == False:
        return None, realy.cpu().numpy()

    for iter, (x, y) in enumerate(dataloader['loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    yhat

    pred = scaler.inverse_transform(yhat) #[t, n, 12]
    
    pred = pred / 3600 
    
    
    return pred.cpu().numpy(), realy.cpu().numpy()



def generate_graph_seq2seq_io_data(data_name, df, x_offsets, y_offsets):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    if data_name == "PEMS04" or data_name == "PEMS08":
        data = df
        num_samples, num_of_vertices, num_of_features = data.shape
        data = data[:, :, [2,]]   #speed, flow, occupy
        x, y = [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
        data_id = []
        for t in range(min_t, max_t):
            x_t = data[t + x_offsets, ...]
            y_t = data[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
            data_id.append(t)
            
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y, data_id
    
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]

    time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
    data_list.append(time_in_day)


    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    data_id = []
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
        data_id.append(t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y, data_id


def load_data(data_name, data_path):
    if data_name == "PEMS04" or data_name == "PEMS08":
        data = np.load(data_path)["data"]
    else:
        data = pd.read_hdf(data_path)
        # 0 is the latest observed sample.
    x_offsets = np.sort(
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    
    x, y, data_id = generate_graph_seq2seq_io_data(
        data_name,
        data,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
    )

    return x, y, data_id