import torch 
import numpy as np
import argparse
import time

from net import gtnet
import pandas as pd
import pickle
from util import metric, StandardScaler, DataLoaderM, load_pickle, load_adj

args = {
    'device': 'cuda:1',
    'data': 'data/METR-LA',
    'adj_data': r'/data/chenjinwen/cjw/RecommendSC/MTGNN/data/sensor_graph/adj_mx.pkl',
        'gcn_true': True,
    'buildA_true': True,
    'load_static_feature': False,
    'cl': True,
    'gcn_depth': 2,
    'num_nodes': 207,
    'dropout': 0.3,
    'subgraph_size': 20,
    'node_dim': 40,
    'dilation_exponential': 1,
    'conv_channels': 32,
    'residual_channels': 32,
    'skip_channels': 64,
    'end_channels': 128,
    'in_dim': 2,
    'seq_in_len': 12,
    'seq_out_len': 12,
    'layers': 3,
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'clip': 5,
    'step_size1': 2500,
    'step_size2': 100,
    
    'epochs': 100,
    'print_every': 50,
    'seed': 101,
    'save': './save/',
    'expid': 1,
    'propalpha': 0.05,
    'tanhalpha': 3,
    'num_split': 1,
    'runs': 10
}


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
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

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

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


def load_data(data_path):
    df = pd.read_hdf(data_path)
        # 0 is the latest observed sample.
    x_offsets = np.sort(
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    
    x, y, data_id = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    return x, y, data_id

if __name__ == "__main__":
    x, y, data_id = load_data(r"/data/chenjinwen/cjw/RecommendSC/MTGNN/data/METR-LA/data.h5")
    dataloader = {
        "x":x,
        "y":y,
        "id":data_id,
    }
    
    scaler = StandardScaler(mean=dataloader['x'][..., 0].mean(), std=dataloader['x'][..., 0].std())

    dataloader["x"][..., 0] = scaler.transform(dataloader['x'][..., 0])

    dataloader['loader'] = DataLoaderM(dataloader['x'], dataloader['y'], 4)
    dataloader['scaler'] = scaler
    
    predefined_A = load_adj(args["adj_data"])
    predefined_A = torch.tensor(predefined_A)-torch.eye(args["num_nodes"])
    predefined_A = predefined_A.to(args['device'])
    
    model = gtnet(args["gcn_true"], args["buildA_true"], args["gcn_depth"], args["num_nodes"], device=args['device'], predefined_A=predefined_A, dropout=args["dropout"], subgraph_size=args["subgraph_size"],
                  node_dim=args["node_dim"],
                  dilation_exponential=args["dilation_exponential"],
                  conv_channels=args["conv_channels"], residual_channels=args["residual_channels"],
                  skip_channels=args["skip_channels"], end_channels= args["end_channels"],
                  seq_length=args["seq_in_len"], in_dim=args["in_dim"], out_dim=args["seq_out_len"],
                  layers=args["layers"], propalpha=args["propalpha"], tanhalpha=args["tanhalpha"], layer_norm_affline=True)
    
    model.load_state_dict(torch.load("/data/chenjinwen/cjw/RecommendSC/MTGNN/save/exp1_0.pth", map_location=torch.device(args['device'])))
    model.to(args['device'])
    
    outputs = []
    realy = torch.Tensor(dataloader['y']).to(args['device'])
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['loader'].get_iterator()):
        testx = torch.Tensor(x).to(args['device'])
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    mae = []
    mape = []
    rmse = []
    for i in range(args["seq_out_len"]):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
    
    
    # dataloader = load_dataset(args["data"], args["batch_size"], args["batch_size"], args["batch_size"])
    # scaler = dataloader['scaler']

    # predefined_A = load_adj(args["adj_data"])
    # predefined_A = torch.tensor(predefined_A)-torch.eye(args["num_nodes"])
    # predefined_A = predefined_A.to(args['device'])
        
    # model = gtnet(args["gcn_true"], args["buildA_true"], args["gcn_depth"], args["num_nodes"], device=args['device'])

    # engine = Trainer(model, args["learning_rate"], args["weight_decay"], args["clip"], args["step_size1"], args["seq_out_len"], ["scaler"], args['device'], args["cl"])
    # #test data
    # outputs = []
    # realy = torch.Tensor(dataloader['y_test']).to(args['device'])
    # realy = realy.transpose(1, 3)[:, 0, :, :]

    # for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    #     testx = torch.Tensor(x).to(args['device'])
    #     testx = testx.transpose(1, 3)
    #     with torch.no_grad():
    #         preds = engine.model(testx)
    #         preds = preds.transpose(1, 3)
    #     outputs.append(preds.squeeze())

    # yhat = torch.cat(outputs, dim=0)
    # yhat = yhat[:realy.size(0), ...]

    # mae = []
    # mape = []
    # rmse = []
    # for i in range(args.seq_out_len):
    #     pred = scaler.inverse_transform(yhat[:, :, i])
    #     real = realy[:, :, i]
    #     metrics = metric(pred, real)
    #     log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    #     print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
    #     mae.append(metrics[0])
    #     mape.append(metrics[1])
    #     rmse.append(metrics[2])