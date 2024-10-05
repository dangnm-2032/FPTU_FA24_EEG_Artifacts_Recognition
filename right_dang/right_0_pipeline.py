import numpy as np
# Importing Pandas Library 
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import *
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from utils import *

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Reshape

from models.EEGNet import *

from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

main_label = 'right'
label_name = ['eyebrows', 'left', 'right', 'both', 'teeth']
trial_num = 10

print(">>>>>>>>> Filtering 5 label with right filter <<<<<<<<<<<<")
os.makedirs(r'.\pipeline_right', exist_ok=True)
for label in label_name:
    os.makedirs(rf'.\pipeline_{main_label}\{label}', exist_ok=True)

    for trial in range(1, trial_num+1):
        raw_df = pd.read_csv(rf'.\raw_data\{label}\{label}_{trial}.csv').drop(columns=['timestamps', 'Right AUX'])

        n_timesteps = 128
        data = raw_df.to_numpy()
        input_data = []
        for i in range(0, data.shape[0] // n_timesteps * n_timesteps, n_timesteps):
            input = data[i:i+n_timesteps].copy()
            for column in range(input.shape[1]):
                x=np.array(input[:, column]) 
                x = filter_right(x)
                input[:, column] = x
            input_data.append(input) 
        input_data = np.concatenate(input_data)

        os.makedirs(rf'.\pipeline_{main_label}\{label}\filtered', exist_ok=True)
        pd.DataFrame(input_data, columns=raw_df.columns).to_csv(rf'.\pipeline_{main_label}\{label}\filtered\{label}_{trial}.csv')
print("--------------- Done ---------------\n")


print(">>>>>>>>> Normalize 5 label with right filter <<<<<<<<<<<<")
os.makedirs(rf'.\pipeline_{main_label}\checkpoints', exist_ok=True)
dfs = []
for label in label_name:
    _path = rf'.\pipeline_{main_label}\{label}'
    # os.makedirs(_path + '\normalized', exist_ok=True)
    for trial in range(1, trial_num+1):
        df = pd.read_csv(_path + rf'\filtered\{label}_{trial}.csv').drop(columns=['Unnamed: 0'])
        dfs.append(df)
dfs = pd.concat(dfs)

scaler = MinMaxScaler()
scaler.fit(dfs)

scaler_filename = rf".\pipeline_{main_label}\checkpoints\scaler.save"
joblib.dump(scaler, scaler_filename) 
print("--------------- Done ---------------\n")


print(">>>>>>>>> Training right <<<<<<<<<<<<")
########### Prepare dataset #########################
raw_data_true = {}
raw_data_false = {}
for label in label_name:
    if label == main_label:
        for trial in range(1, trial_num + 1):
            raw_data_true[len(raw_data_true)] = [
                rf'.\raw_data\{label}\{label}_{trial}.csv',
                rf'.\roi_v2\{label}\{label}_{trial}.csv'
            ]
    
    else:
        c1, c2 = np.random.choice(range(1, trial_num+1), 2)
        raw_data_false[len(raw_data_false)] = [
            rf'.\raw_data\{label}\{label}_{c1}.csv',
            rf'.\roi_v2\{label}\{label}_{c1}.csv'
        ]
        raw_data_false[len(raw_data_false)] = [
            rf'.\raw_data\{label}\{label}_{c2}.csv',
            rf'.\roi_v2\{label}\{label}_{c2}.csv'
        ]
#####################################################

############# LOAD SCALER ###########################
scaler = joblib.load(rf".\pipeline_{main_label}\checkpoints\scaler.save")
#####################################################

################### Split - filter - normalize ###################
dataset_true = {}
for label_ in raw_data_true:
    data, label = process_raw_record(raw_data_true[label_])

    dataset_true[label_] = {}
    temp_data, temp_label = create_dataset(data, label, filter_right, scaler)
    print(temp_data.shape, temp_label.shape)
    temp_data, temp_label = unison_shuffled_copies(temp_data, temp_label)
    train_idx = int(temp_data.shape[0] * 0.8)
    dataset_true[label_]['train_data'] = temp_data[:train_idx]
    dataset_true[label_]['train_label'] = temp_label[:train_idx]
    dataset_true[label_]['test_data'] = temp_data[train_idx:]
    dataset_true[label_]['test_label'] = temp_label[train_idx:]

    print(
        label_, 
        dataset_true[label_]['train_data'].shape,
        dataset_true[label_]['train_label'].shape,
        dataset_true[label_]['test_data'].shape,
        dataset_true[label_]['test_label'].shape,
        sep=' --- '
    )

dataset_false = {}
for label_ in raw_data_false:
    data, label = process_raw_record(raw_data_false[label_])

    dataset_false[label_] = {}
    temp_data, temp_label = create_dataset(data, label, filter_right, scaler, epsilon=0.1)
    temp_label[temp_label == 1] = 0
    temp_data, temp_label = unison_shuffled_copies(temp_data, temp_label)
    train_idx = int(temp_data.shape[0] * 0.8)
    dataset_false[label_]['train_data'] = temp_data[:train_idx]
    dataset_false[label_]['train_label'] = temp_label[:train_idx]
    dataset_false[label_]['test_data'] = temp_data[train_idx:]
    dataset_false[label_]['test_label'] = temp_label[train_idx:]

    print(
        label_, 
        dataset_false[label_]['train_data'].shape,
        dataset_false[label_]['train_label'].shape,
        dataset_false[label_]['test_data'].shape,
        dataset_false[label_]['test_label'].shape,
        sep=' --- '
    )
####################################################################

############# CONCATENATE DATASET ##################################
train_x = []
train_y = []
test_x = []
test_y = []

for label in dataset_true:
    train_x.append(dataset_true[label]['train_data'])
    train_y.append(dataset_true[label]['train_label'])
    test_x.append(dataset_true[label]['test_data'])
    test_y.append(dataset_true[label]['test_label'])
for label in dataset_false:
    train_x.append(dataset_false[label]['train_data'])
    train_y.append(dataset_false[label]['train_label'])
    test_x.append(dataset_false[label]['test_data'])
    test_y.append(dataset_false[label]['test_label'])

train_x = np.concatenate(train_x)
train_y = np.concatenate(train_y)
test_x = np.concatenate(test_x)
test_y = np.concatenate(test_y)

train_x = train_x.transpose((0, 2, 1))
test_x = test_x.transpose((0, 2, 1))

train_x = np.expand_dims(train_x, axis=-1)
test_x = np.expand_dims(test_x, axis=-1)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

train_x, train_y = unison_shuffled_copies(train_x, train_y)
####################################################################

############# MODELING ##########################
base_model = EEGNet_SSVEP(
        nb_classes=1, Chans = 4, Samples = 128, 
    dropoutRate = 0.5, kernLength = 40, F1 = 10, 
    D = 2, F2 = 10, dropoutType = 'Dropout')

x = base_model.layers[-3].output
x = Dense(128*2, activation='relu')(x)
x = Reshape((128, 2))(x)
x = Activation('softmax', name = 'softmax')(x)
model = Model(inputs=base_model.input, outputs=x)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy', 
    optimizer='adam',
    metrics=['accuracy']
)
#################################################

######### TRAINING ##############################
history = model.fit(
    train_x, 
    train_y,
    epochs=20,
    validation_data=(test_x, test_y),
)
#################################################

############### RESULT ##########################
plt.figure(figsize=(20, 10)).suptitle("All labels")
plt.subplot(121)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(122)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

os.makedirs(rf'pipeline_{main_label}\results', exist_ok=True)
plt.savefig(rf'pipeline_{main_label}\results\training_result.jpg')

model.save(rf'pipeline_{main_label}\checkpoints\checkpoint.keras')

y_pred = model.predict(test_x)
y_true = test_y
y_pred = np.argmax(y_pred, 2)


cm_total = np.zeros((2, 2))

for y_t, y_p in zip(y_true, y_pred):
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    cm = np.array(cm)
    cm_total = cm_total + cm


result = []
for cls in range(2):
    tp = cm_total[cls, cls]
    fn = np.sum(np.delete(cm_total[cls, :], cls))
    fp = np.sum(np.delete(cm_total[:, cls], cls))
    tn = np.delete(cm_total, cls, axis=0)
    tn = np.sum(np.delete(tn, cls, axis=1))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    acc = (tp + tn) / (tp + fn + tn + fp)
    specifity = tn/(tn+fp)

    result.append([precision, recall, f1, acc, specifity])

result = np.array(result)
print(f'precision, recall, f1, acc, specifity\n{result}')
plt.figure(figsize=(20, 10))
plt.title("Confusion Matrix of All labels Detection Model")
plt.matshow(result, fignum=False)
plt.xticks([0, 1, 2, 3, 4], ['Positive Predictive\nValue (Precision)', 'True Positive\nRate (Recall)', 'F1 Score', 'Accuracy', 'True Negative\nRate (Specifity)'])
plt.yticks([0, 1], ['Not command', 'Is command'])
plt.xlabel("Metric")
plt.ylabel("Class")
for (i, j), z in np.ndenumerate(result):
    plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
plt.colorbar()
plt.savefig(rf'pipeline_{main_label}\results\confussion_matrix.jpg')
#################################################
print("--------------- Done ---------------\n")

print(">>>>>>>>> INFERENCE OFFLINE <<<<<<<<<<<<")
df_eyebrows = pd.read_csv(r'.\inference_data\eyebrows.csv').drop(columns=['timestamps', 'Right AUX'])
df_left = pd.read_csv(r'.\inference_data\left.csv').drop(columns=['timestamps', 'Right AUX'])
df_right = pd.read_csv(r'.\inference_data\right.csv').drop(columns=['timestamps', 'Right AUX'])
df_both = pd.read_csv(r'.\inference_data\both.csv').drop(columns=['timestamps', 'Right AUX'])
df_teeth = pd.read_csv(r'.\inference_data\teeth.csv').drop(columns=['timestamps', 'Right AUX'])

data, input_data = get_input(df_eyebrows, filter_right, scaler)
y_pred_onehot = get_output(input_data, model)
plot_data_result(
    data, 
    y_pred_onehot, 
    "Eyebrows data",
    rf'pipeline_{main_label}\results\inference_eyebrows.jpg'
)

data, input_data = get_input(df_right, filter_right, scaler)
y_pred_onehot = get_output(input_data, model)
plot_data_result(
    data, y_pred_onehot, "Right data",
    rf'pipeline_{main_label}\results\inference_right.jpg'
)

data, input_data = get_input(df_left, filter_right, scaler)
y_pred_onehot = get_output(input_data, model)
plot_data_result(
    data, y_pred_onehot, "Left data",
    rf'pipeline_{main_label}\results\inference_left.jpg'
)

data, input_data = get_input(df_both, filter_right, scaler)
y_pred_onehot = get_output(input_data, model)
plot_data_result(
    data, y_pred_onehot, "Both data",
    rf'pipeline_{main_label}\results\inference_both.jpg'
)

data, input_data = get_input(df_teeth, filter_right, scaler)
y_pred_onehot = get_output(input_data, model)
plot_data_result(
    data, y_pred_onehot, "Teeth data",
    rf'pipeline_{main_label}\results\inference_teeth.jpg'
)
print("--------------- Done ---------------\n")