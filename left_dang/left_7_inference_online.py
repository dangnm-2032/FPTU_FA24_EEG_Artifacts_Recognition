import numpy as np  # Module that simplifies computations on matrices
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

from models.EEGNet import *

from preprocessing import *

import warnings
warnings.filterwarnings("ignore")


def pipeline(x, filter, scaler):
    x_new = x.copy()
    for col in range(x_new.shape[1]):
        x_new[:, col] = filter(x_new[:, col])
    x_new = scaler.transform(x_new)

    return x_new


##################### MODEL ###############################################
model = load_model(r'.\pipeline_left\checkpoints\checkpoint.keras')
###########################################################################


##################### SCALER ##############################################
import joblib
scaler = joblib.load(r'.\pipeline_left\checkpoints\scaler.save')  
###########################################################################


##################### CONNECT EEG STREAM ##################################
print('Looking for an EEG stream...')
streams = resolve_byprop('type', 'EEG', timeout=2)
if len(streams) == 0:
    raise RuntimeError('Can\'t find EEG stream.')
print("Start acquiring data")
inlet = StreamInlet(streams[0], max_chunklen=128)
eeg_time_correction = inlet.time_correction()
###########################################################################


n_timesteps = 128
while True:
    # inlet.pull_chunk(timeout=1.0, max_samples=128)
    eeg_data, timestamp = inlet.pull_chunk(timeout=1.0, max_samples=n_timesteps)
    eeg_data = np.array(eeg_data)[:, :-1]  # sample, channel

    ########################### Preprocessing ###################################
    x = pipeline(eeg_data, filter_left, scaler)
    
    input = np.concatenate([x], axis=1)
    input = np.expand_dims(input, 0)
    input = np.expand_dims(input, -1)
    input = input.transpose(0, 2, 1, 3)
    # print(input.shape)
    assert input.shape == (1, 4, 128, 1)
    #############################################################################


    ############################### Inference ###################################
    y_pred = model.predict(input)
    y_pred = np.argmax(y_pred, 2)[0]
    #############################################################################
    print(y_pred)

    # 1 eyebrows 2 teeth 3 right
    # pred = np.max(y_pred)
    # print(pred)

    # if pred == 1:
    #     print('Left')
    #     ser.write(bytes('3', 'ascii'))
    # elif pred == 2:
    #     print('Right')
    #     ser.write(bytes('4', 'ascii'))
    # elif pred == 0:
    #     print('Stop')
    #     ser.write(bytes('0', 'ascii'))