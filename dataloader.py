import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import random
import pandas as pd
import torch
from utils import Task, Worker

vehicle_p = [
    [   [0.00000000,-0.03341761, 5.10983452, -0.00000010, 0.00187154,-0.52883091,37.50573903,0,],
        [0.00003856,-0.00858022,0.57734626,0.00000000,0.00000000,0.00000000,5.43052047,0],
        [0.00000000,0.00000000,2.87000000,0.00000000,0.00000000,0.00000000,	1000.00000000, 0.0]],
     
    [   [-0.00000089,	0.00275863,	0.35480439,	3.43785781,	0.00226692,	0.18808284,	1.12145525, 0.0],
        [-0.00006237,	0.01084828,	-0.16587581,	0.00000000,	0.00000000,	0.00000000,	16.58758101, 0.0],
        [0.00000000,	0.00000000,	2.87000000,	0.00000000,	0.00000000,	0.00000000,	1000.00000000, 0.0],
    ],
    [   [0.00008498,	-0.01272809,	0.99900386,	0.88246968,	0.00198237,	0.04202091,	2.87682286, 0.0],
        [0.00056237,	-0.07639736,	4.19882320,	0.00000000,	0.00000000,	0.00000000,	3.78955163, 0.0],
        [0.00000000,	0.00000000,	1.10000000,	0.00000000,	0.00000000,	0.00000000,	1000.00000000, 0.0],
    ]]
vehicle_p = np.array(vehicle_p)

def pre_process_data(data_path, output_path, graph_save_path, num_task, task_path):
    df = pd.read_csv(data_path)

    # 转换 utcTimestamp 为 datetime 格式
    df['utcTimestamp'] = pd.to_datetime(df['utcTimestamp'], format='%a %b %d %H:%M:%S %z %Y')
    # 增加归一化时间列
    df['normalized_time'] = (df['utcTimestamp'].dt.hour * 3600 + 
                             df['utcTimestamp'].dt.minute * 60 + 
                             df['utcTimestamp'].dt.second) / 86400  # 归一化到 [0, 1] 范围
    
    df['date'] = df['utcTimestamp'].dt.date

    # 创建 venueId 到索引的映射
    id_to_idx = {id_: idx for idx, id_ in enumerate(df['venueId'].unique())}
    
    df['poiId'] = df['venueId'].map(id_to_idx)
    
    transport_models = {}
    for user_id in df['userId'].to_list():
        transport_model = [1, 2, 3]
        random.shuffle(transport_model)
        transport_models[user_id] = transport_model
    
    def random_transport(x):
        rd = random.random()
        if rd < 0.1:
            return transport_models[x][0]
        elif rd < 0.3:
            return transport_models[x][1]
        else:
            return transport_models[x][2]
        
    df["transportation"] = df['userId'].apply(lambda x: random_transport(x))
    # 创建 trajectoryId
    df['trajectoryId'] = df.groupby(['userId', 'date']).ngroup()
    #df['trajectoryId'] = df['trajectoryId'].apply(lambda x: f"{df['userId'][x]}_{x}")

    # 删除不需要的列
    df = df.drop(columns=["venueId", "venueCategory", "venueCategoryId", "timezoneOffset"])
    poi_num = max(df['poiId'].unique()) + 1
    X = np.zeros((poi_num, 4))
    A = np.zeros((poi_num, poi_num))
    for trajId in df['trajectoryId'].unique():
        trajdf = df[df['trajectoryId'] == trajId]
        trajdf = trajdf[["poiId", "normalized_time", "latitude", "longitude"]]
        trajlist = list(trajdf.itertuples(index=False, name=None))
        trajlist.sort(key=lambda x:x[1])
        for idx, poi in enumerate(trajlist):
            if idx > 0:
                A[trajlist[idx-1][0]][trajlist[idx][0]] += 1
            X[trajlist[idx][0]][0] =  poi[2]
            X[trajlist[idx][0]][1] =  poi[3]
            X[trajlist[idx][0]][2] += 1
            X[trajlist[idx-1][0]][3] += 1
            
    print(np.sum(A)/(np.sum(A > 0)))
    # 将邻接矩阵转换为 edge_index 格式
    adj = A / np.max(A)
    row, col = np.nonzero(adj)
    edge_index = np.array([row, col])
    edge_wieight = adj[row, col]
    
    np.save("./data/NYC/graph_adj.npy", edge_index)
    np.save("./data/NYC/graph_weight.npy", edge_wieight)

    # # np.nonzero(A) 返回的是非零的边的索引对
    # row, col = np.nonzero(A)
    # # 提取相应的权重
    # edge_weight = A[row, col]

    # # 将 numpy 数据转换为 torch 张量
    # edge_index = torch.tensor([row, col], dtype=torch.long)
    # edge_attr = torch.tensor(edge_weight, dtype=torch.float)  # 边的权重（edge_attr）
    # x = torch.tensor(X, dtype=torch.float)

    # # 创建 PyG 的 Data 对象，并包括边权重信息
    # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # # 保存 Data 对象为文件
    # torch.save(data, graph_save_path)
    
    # 生成随机任务
    df['utcTimestamp'] = df['utcTimestamp'].astype('int64') // 10**9
    time_min = min(df['utcTimestamp'])
    time_max = max(df['utcTimestamp'])
    tasks = []
    for _ in range(num_task):
        poi_id = np.random.randint(0, poi_num)
        task_time = np.random.randint(time_min, time_max)
        tasks.append([poi_id, task_time])
    tasks.sort(key=lambda x:x[1])
    
    
    with open(task_path+f"{num_task}task.txt", 'w') as f:
        for id, tim in tasks:
            f.write(f"{tim} {id}\n")
    
    # 保存到 CSV
    df.to_csv(output_path, index=False)


class TrajectoryDatasetTrain(Dataset):
    def __init__(self, train_df, trainset):
        self.df = train_df
        self.traj_seqs = []  # traj id: user id + traj no.
        self.input_seqs = []
        self.label_seqs = []

        for traj_id in tqdm(set(train_df['trajectoryId'].tolist())):
            traj_df = train_df[train_df['trajectoryId'] == traj_id]
            user_id = traj_df["userId"].to_list()[0]
            if user_id not in trainset:
                continue
            
            poi_idxs = traj_df['poiId'].to_list()
            time_feature = traj_df['normalized_time'].to_list()
            transportation = traj_df['transportation'].to_list()
            
            poi_idxs = [x for x, _ in sorted(zip(poi_idxs, time_feature))]
            time_feature = sorted(time_feature)
            transportation = [x for x, _ in sorted(zip(transportation, time_feature))]

            input_seq = []
            label_seq = []
            
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_feature[i], transportation[i]))
                label_seq.append((poi_idxs[i + 1], time_feature[i + 1], transportation[i+1]))

            self.traj_seqs.append(traj_id)
            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)

    def __len__(self):
        assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

class TrajectoryDatasetVal(Dataset):
    def __init__(self, df, valset):
        self.df = df
        self.traj_seqs = []
        self.input_seqs = []
        self.label_seqs = []

        for traj_id in tqdm(set(df['trajectoryId'].tolist())):
            # Ger POIs idx in this trajectory
            traj_df = df[df['trajectoryId'] == traj_id]
            user_id = traj_df.tolist()[0]
            
            if user_id not in valset:
                continue
            
            poi_idxs = traj_df['poiId'].to_list()
            time_feature = traj_df['normalized_time'].to_list()
            transportation = traj_df['transportation'].to_list()

            # Construct input seq and label seq
            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_feature[i], transportation[i]))
                label_seq.append((poi_idxs[i + 1], time_feature[i + 1], transportation[i+1]))

            # # Ignore seq if too short
            # if len(input_seq) < args.short_traj_thres:
            #     continue

            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)
            self.traj_seqs.append(traj_id)

    def __len__(self):
        assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])
    
    
def readData(path:str, num_task:int, num_worker):
    '''
    return workers, tasks, graph_adj, graph_weight, poi_position_array, edges
    '''
    if path == "gMission":
        path = "./data/gMission/gMission_cap1.txt"
        tasks = [] 
        workers = []
        
        with open(path, 'r') as f:
            content = f.readlines()
            for d in content[1:]:
                d = d.split()
                if d[1] != "t":
                    workers.append(np.array([float(x) for x in d[0:1] + d[2:6]])) #[time, x, y, range, capacity, deadline]
                else:
                    tasks.append(np.array([float(x) for x in d[0:1] + d[2:]]))#[time, x, y, deadline, payoff]
        
        tasks = np.array(tasks)
        worker = np.array(workers)
            
        return tasks, worker
    elif path == "NYC" or "TKY":
        data_path = f"./data/{path}/{path}.csv"
        df = pd.read_csv(data_path)
        graph_adj = np.load(f"./data/{path}/graph_adj.npy")
        graph_weight = np.load(f"./data/{path}/graph_weight.npy")
        workers = []
        tasks = []
        poi_position = {}
        poi_position_list = []
        
        len_poi = 0
        for poi_id, group in df.groupby("poiId"):
            poi_position[poi_id] = [group.iloc[0]["latitude"], group.iloc[0]["longitude"]]
            len_poi = max(len_poi, poi_id)
            
        for i in range(len_poi+1):
            poi_position_list.append(poi_position[i]) 
            
        #构建路网
        poi_position_array = np.array(poi_position_list)
        n = len(poi_position_array)
        if os.path.exists(f"./data/{path}/road.json"):
            # 读取 JSON 文件
            with open(f"./data/{path}/road.json", 'r') as f:
                edges_data = json.load(f)

            # 恢复边列表
            edges = [(edge['Node1'], edge['Node2'], edge['Distance']) for edge in edges_data]
        else:
            edges = []
            # 遍历每个点，逐行计算距离，避免大矩阵的内存分配
            for i in range(n):
                distances = np.linalg.norm(poi_position_array - poi_position_array[i], axis=1)
                sorted_indices = np.argsort(distances)[1:3]  # 选择最近的两个点
                for j in sorted_indices:
                    edges.append((i, j, distances[j]))  # (节点1, 节点2, 距离)
            edges_data = [{'Node1': int(edge[0]), 'Node2': int(edge[1]), 'Distance': float(edge[2])} for edge in edges]

            with open(f"./data/{path}/road.json", 'w') as f:
                json.dump(edges_data, f, indent=4)
        
        for traj_id in tqdm(set(df['trajectoryId'].tolist())):
            traj_df = df[df['trajectoryId'] == traj_id]
            user_id = traj_df["userId"].to_list()[0]
            
            traj_df_sort = traj_df.sort_values(by=['utcTimestamp'])
            traj_df_sort = traj_df_sort.to_dict(orient='list')
            
            # df['utcTimestamp'] = df['utcTimestamp'].astype('int64') // 10**9

            poi_list = []
            transp_list = []
            
            for i in range(len(traj_df_sort["poiId"])):
                poi_list.append(traj_df_sort["poiId"][i])
                transp_list.append(traj_df_sort["transportation"][i])
                workers.append(Worker(traj_df_sort['utcTimestamp'][i], traj_df_sort["latitude"][i], traj_df_sort["longitude"][i], 300, traj_df_sort["poiId"][i], poi_list, 3, transp_list=transp_list)) 
        #workers = random.sample(workers, num_worker)
            if len(workers) > num_worker:
                break
        with open(f"./data/{path}/{num_task}task.txt", 'r') as f:
            content = f.readlines()
            for d in content:
                d = d.split()
                poi_id = int(d[1])
                tasks.append(Task(int(d[0]), poi_position[poi_id][0], poi_position[poi_id][1], 300, poi_id))
        return workers, tasks, graph_adj, graph_weight, poi_position_array, edges

def getrandomData(data_name, num_task, num_worker, args):
    workers = []
    tasks = []
    
    root_path = "/data/chenjinwen/cjw/RecommendSC/MTGNN/"
    
    if data_name == "PEMS04" or data_name == "PEMS08":
        data = np.load(root_path + f"data/{data_name}/pems{data_name[-2:]}.npz")["data"]
        num_samples, num_nodes, num_of_features = data.shape
        poi_position = np.zeros((num_nodes, num_nodes))
        time_list = list(range(0, num_samples))
        time_list = [x * 300 for x in time_list]
        
    else:
        df = pd.read_hdf(root_path + f"data/{data_name}/data.h5")
        poi_position = pd.read_csv(root_path +f"data/{data_name}/sensor_locations.csv")
        sensor_id = list(df.columns)
        
        poi_position = poi_position[["latitude", "longitude"]].values
        
        num_samples, num_nodes = df.shape
        
        time_list = df.index.values.astype('int64') // 10**9

    min_time = np.min(time_list)
    max_time = np.max(time_list)
    
    for i in range(num_worker):
        poiId = random.randint(0, num_nodes-1)
        hispoi = [poiId, ]
        publish_time = random.randint(min_time, max_time)
        for _ in range(5):
            dis = np.linalg.norm(poi_position - poi_position[hispoi[0]], axis=1)
            sorted_indices = np.argsort(dis)[1:3]
            temp_id = np.random.choice(sorted_indices)
            hispoi.insert(0, temp_id)
        tp = vehicle_p[random.randint(0, 2)]
        workers.append(Worker(publish_time, poi_position[poiId][0], poi_position[poiId][1], args["wdead"], poiId, hispoi, 3, transp_list = random.randint(0,2), p=tp))
    
    if os.path.exists(root_path + f"data/{data_name}/{num_task}task.txt"):
        with open(root_path + f"data/{data_name}/{num_task}task.txt", 'r') as f:
            content = f.readlines()
            for d in content:
                d = d.split()
                poi_id = int(d[1])
                tasks.append(Task(int(d[0]), poi_position[poi_id][0], poi_position[poi_id][1], args["sdead"], poi_id))
    else:
        taskstxt = []
        for _ in range(num_task):
            poi_id = np.random.randint(0, num_nodes-1)
            task_time = np.random.randint(min_time, max_time)
            taskstxt.append([poi_id, task_time])
        taskstxt.sort(key=lambda x:x[1])
        
        with open(root_path + f"data/{data_name}/{num_task}task.txt", 'w') as f:
            for id, tim in taskstxt:
                f.write(f"{tim} {id}\n")
        with open(root_path + f"data/{data_name}/{num_task}task.txt", 'r') as f:
            content = f.readlines()
            for d in content:
                d = d.split()
                poi_id = int(d[1])
                tasks.append(Task(int(d[0]), poi_position[poi_id][0], poi_position[poi_id][1], 300, poi_id))
                
                
    # generate map
    if data_name == "PEMS04" or data_name == "PEMS08":
        adj = pd.read_csv(root_path + f"data/{data_name}/distance.csv")
        poi_map = np.zeros((num_nodes, num_nodes))
        
        for _, (x, y, c) in adj.iterrows():
            poi_map[int(x), int(y)] = c/1000
            poi_map[int(y), int(x)] = c/1000
        poi_map[poi_map==0] = 1e9
    
    else:    
        n = len(poi_position)
        poi_map = np.zeros((n,n))
        for i in range(n):
            dis = np.linalg.norm(poi_position - poi_position[i], axis=1)
            dis_id = np.argsort(dis)
            for j in dis_id[:6]:
                poi_map[i][j] = dis[j] * 111
                poi_map[j][i] = dis[j] * 111
        poi_map[poi_map==0] = 1e9
    
    return workers, tasks, poi_map, time_list
        
    
        

if __name__ == "__main__":
    # pre_process_data("./data/NYC/dataset_TSMC2014_NYC.csv","./data/NYC/NYC.csv", "./data/NYC/NYCgraph.pt", 3000, "./data/NYC/")
    # df = pd.read_csv("./data/NYC/NYC.csv")
    
    # user_list = list(set(df['userId'].to_list()))
    # random.shuffle(user_list)
    # user_list = [str(x)+' ' for x in user_list]
    # with open('./data/NYC/userIdList.txt', 'w') as f:
    #     f.writelines(user_list)
    # data = []
    # with open('./data/NYC/NYCuserIdList.txt', 'r') as f:
    #     data = f.readline().strip().split(' ')
    #     data = [int(x) for x in data]
    
    # trainset = data[:int(len(data)*0.8)]
    # valset = data[int(len(data)*0.8):]
    # train_dataset = TrajectoryDatasetTrain(df, trainset)
    
    # workers, tasks, graph_adj, graph_weight, poi_position_list = readData("NYC", 3000, 1000)
    workers, tasks, poi_map, time_list = getrandomData("PEMS04", 1000, 3000)