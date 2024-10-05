import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def process_raw_record(args):
    input_path = args[0]
    label_path = args[1]

    input_df = pd.read_csv(input_path).drop(columns=['timestamps', 'Right AUX'])
    input_data = input_df.to_numpy()

    label_df = pd.read_csv(label_path).to_numpy()
    label = np.zeros(len(input_data))
    
    for row in label_df:
        label[row[0]:row[1]] = row[2]

    return input_data, label


def pipeline(x, filter, scaler, i, time_step):
    x_new = x[i:i+time_step].copy()
    for col in range(x_new.shape[1]):
        x_new[:, col] = filter(x_new[:, col])
    x_new = scaler.transform(x_new)

    return x_new


def create_dataset(x, y, filter, scaler, time_step=128, epsilon=0):
    assert x.shape[0] == y.shape[0]
    x_new = []
    y_new = []

    for i in range(0, x.shape[0] - time_step):
        if 1 in y[i:i+time_step] or np.random.random() < epsilon:
            x_new.append(pipeline(x, filter, scaler, i, time_step))
            y_new.append(y[i:i+time_step])
    
    return np.array(x_new), np.array(y_new)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_input(df, filter, scaler):
    # Split then filtered
    n_timesteps = 128
    data = df.to_numpy()
    input_data = []
    for i in range(0, data.shape[0] // n_timesteps * n_timesteps, n_timesteps):
        
        x_eyebrows = pipeline(data, filter, scaler, i, n_timesteps)

        input = np.concatenate([
            # x_left, x_right, x_both, x_teeth, 
            x_eyebrows
        ], axis=1)
        input_data.append(input)    

    input_data = np.array(input_data)
    input_data = input_data[:, :, :, np.newaxis]
    input_data = input_data.transpose(0, 2, 1, 3)
    print(input_data.shape)
    return data, input_data

def get_output(input_data, model):
    y_pred = model.predict(input_data)
    y_pred = np.argmax(y_pred, axis=2)
    y_pred = np.concatenate(y_pred, axis=0)
    y_pred_onehot = np.zeros((y_pred.size, 2))
    y_pred_onehot[np.arange(y_pred.size), y_pred] = 1

    return y_pred_onehot

def plot_data_result(data, y_pred_onehot, title, path):
    plt.figure(figsize=(50, 30)).suptitle(title, fontsize=40)
    plt.subplot(6, 1, 1)
    plt.title("TP9", fontsize=40)
    plt.plot(data[:, 0])
    plt.subplot(6, 1, 2)
    plt.title("AF7", fontsize=40)
    plt.plot(data[:, 1])
    plt.subplot(6, 1, 3)
    plt.title("AF8", fontsize=40)
    plt.plot(data[:, 2])
    plt.subplot(6, 1, 4)
    plt.title("TP10", fontsize=40)
    plt.plot(data[:, 3])
    plt.subplot(6, 1, 5)
    plt.title("Result", fontsize=40)
    plt.plot(y_pred_onehot[:, 1])
    plt.savefig(path)