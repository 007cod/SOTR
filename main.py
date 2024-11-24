import numpy as np
import time
import os 
import pandas as pd
import random
import torch
from utils import metric, Worker, Task
from assignment import SRTS
from dataloader import readData, getrandomData
from get_pref import pref, get_trans_model, get_emission, pred_speed, load_data
from SOTR.util import StandardScaler
from SOTR.net import gtnet, GCNet
from SOTR.waveNet import wavenet
import yaml

np.random.seed(0)
random.seed(0)

def init(args):
    
    # workers, tasks, graph_adj, graph_weight, poi_position_array, edges = readData(args["data_name"], args["task_num"], args["worker_num"])
    # pref_matrix = pref(workers, tasks, graph_adj, graph_weight)
    # transport_model = get_trans_model(workers, tasks)
    train_path = "/data/chenjinwen/cjw/RecommendSC/SOTR/"
    if args["data_name"] == "PEMS04" or args["data_name"] == "PEMS08":
        data_path = train_path + f"data/{args['data_name']}/pems{args['data_name'][-2:]}.npz"
    else:
        data_path = f"/data/chenjinwen/cjw/RecommendSC/SOTR/data/{args['data_name']}/data.h5"

    #初始化模型和数据
    x, y, data_id = load_data(args['data_name'], data_path)
    dataloader = {
        "x":x,
        "y":y,
        "id":data_id,
    }
    print("load model")
    scaler = StandardScaler(mean=dataloader['x'][..., 0].mean(), std=dataloader['x'][..., 0].std())
    if args["model"] =="GCNet":
        model = GCNet(args["gcn_true"], args["buildA_true"], args["gcn_depth"], args["num_nodes"],
                device = args["device"], dropout=args["dropout"], subgraph_size=args["subgraph_size"],
                node_dim=args["node_dim"],conv_channels=args["conv_channels"], end_channels= args["end_channels"],
                seq_length=args["seq_in_len"], in_dim=args["in_dim"], out_dim=args["seq_out_len"],
                layers=args["layers"], propalpha=args["propalpha"], tanhalpha=args["tanhalpha"])
    elif args["model"] == "wavenet":
        model = wavenet(args["device"], args["num_nodes"], args["dropout"], in_dim=args["in_dim"],out_dim=args["seq_out_len"])
    else:
        model = gtnet(args["gcn_true"], args["buildA_true"], args["gcn_depth"], args["num_nodes"], device=args['device'], dropout=args["dropout"], subgraph_size=args           ["subgraph_size"],node_dim=args["node_dim"],
                    dilation_exponential=args["dilation_exponential"],
                    conv_channels=args["conv_channels"], residual_channels=args["residual_channels"],
                    skip_channels=args["skip_channels"], end_channels= args["end_channels"],
                    seq_length=args["seq_in_len"], in_dim=args["in_dim"], out_dim=args["seq_out_len"],
                    layers=args["layers"], propalpha=args["propalpha"], tanhalpha=args["tanhalpha"], layer_norm_affline=True)

    model.load_state_dict(torch.load(f"./SOTR/save/{args['data_name']}/{args['model']}exp1_0.pth", map_location=torch.device(args['device']), weights_only=True))
    model.to(args['device'])

    #获取预测速度[t, n, 12]
    print("get speed km/s")
    speed_pred, speed_real = pred_speed(model, dataloader, scaler, pred_speed = args["pred_speed"], device=args['device'])  
    
    return speed_pred, speed_real

def main(args, speed_pred, speed_real):
    print(f"loading data {args['data_name']}")
    workers, tasks, poi_map, time_list = getrandomData(args["data_name"], args["task_num"], args["worker_num"], args)
    pref_matrix = np.random.uniform(0.01, 0.7, size=(len(workers), len(tasks)))
    
    print("get emission")
    emission, time_cost, emission_real, dis  = get_emission(workers, tasks, poi_map, speed_pred, speed_real, time_list, pred_speed = args["pred_speed"])
    
    f = open(args['output_path'], 'a')
    methods = ["TopK", "MTR",  "GCA", "STR", "STRS"]
    
    for md in methods:  
        print(f"assign {md}")
        assign_model = SRTS(tasks, workers, pref_matrix, emission, dis, args, time_cost=time_cost)
        t1 = time.time()
        recommed_task_seqs = assign_model.assign(name=md)
        print(f"{md} time: {time.time() - t1}")
        cputime = time.time() - t1

        profit, cost, apu = metric(recommed_task_seqs, pref_matrix, emission, assign_model.RS,  k=args["k"])
        _, real_cost, apu = metric(recommed_task_seqs, pref_matrix, emission_real,assign_model.RS, k=args["k"])
        print(f"{md} profit:{profit}, cost:{cost}, real cost:{real_cost}")
        
        if args["test_name"] == "wdead" or args["test_name"] == "sdead":
            f.write(f'Data:{args["data_name"]} model:{args["model"]} assign:{md} {args["test_name"]}:{args[args["test_name"]]/3600} profit:{profit:.2f} cost:{real_cost/profit:.2f} realcost:{real_cost/profit:.2f} time:{cputime:.2f} apu:{apu:.2f}\n')
        else:
            f.write(f'Data:{args["data_name"]} model:{args["model"]} assign:{md} {args["test_name"]}:{args[args["test_name"]]} profit:{profit:.2f} cost:{real_cost/profit:.2f} realcost:{real_cost/profit:.2f} time:{cputime:.2f} apu:{apu:.2f}\n')
    

if __name__ == "__main__":
    args = {
    "data_name":"PEMS04",  #PEMS-BAY, PEMS04
    "task_num":3000,
    "worker_num":1000,
    "wdead":3600,
    "sdead":3600,
    "model":"GCNet",
    "model_path":"",
    # "transport_cost":[0.1, 0.5, 0.8],  #摩托车，轻型卡车，汽车
    "range":3,
    "v":1,
    "iter_num":1e5,
    "k":5,
    "max_iter_num":20,
    "KMcost":True,
    "pred_speed":True,
    "device":'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    for data_name in ["PEMS08", "PEMS04"]:
        args["data_name"] = data_name
        args["output_path"] =  f"./output/{data_name}/w.txt"
        if not os.path.exists(f"./output/{data_name}/"):
            os.mkdir(f"./output/{data_name}/")
        with open(f'./SOTR/model_config/model_config_{args["data_name"]}.yaml', 'r', encoding='utf-8') as file:
            model_args = yaml.safe_load(file)
        args.update(model_args)
        speed_pred, speed_real = init(args)
        main(args, speed_pred, speed_real)
                