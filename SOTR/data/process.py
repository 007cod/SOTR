import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(data_path, x_offsets, y_offsets):
    data = np.load(data_path)["data"]   # (sequence_length, num_of_vertices, num_of_features)
    # data = data[:, :, -1]
    num_samples, num_of_vertices, num_of_features = data.shape
    data = data[:, :, [2, ]]   #speed, flow, occupy
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(data_path, output_dir):
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        data_path,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

if __name__ == "__main__":
    generate_train_val_test("/data/chenjinwen/cjw/RecommendSC/MTGNN/data/PEMS04/pems04.npz", "/data/chenjinwen/cjw/RecommendSC/MTGNN/data/PEMS04/")
    generate_train_val_test("/data/chenjinwen/cjw/RecommendSC/MTGNN/data/PEMS08/pems08.npz", "/data/chenjinwen/cjw/RecommendSC/MTGNN/data/PEMS08/")