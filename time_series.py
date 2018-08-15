import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, LSTM

file_name = 'profiles/southern-cairngorms.csv'
# file_name = 'profiles/northern-cairngorms.csv'
# file_name = 'profiles/glencoe.csv'
# file_name = 'profiles/torridon.csv'
# file_name = 'profiles/lochaber.csv'
# file_name = 'profiles/creag-meagaidh.csv'

observed_hazard = 'Observed aval. hazard'
temp_gradient = 'Max Temp Grad'
hardness_gradient = 'Max Hardness Grad'
snow_depth = 'Total Snow Depth'

snow_temp = 'Snow Temp'
no_settle = 'No Settle'
insolation = 'Insolation'
foot_pen = 'Foot Pen'
ski_pen = 'Ski Pen'
drift = 'Drift'

features = [observed_hazard, temp_gradient, hardness_gradient, snow_depth]
feature_count = len(features)
look_back = 7

def numerical_labels(data):
    if data == 'Low':
        return 0
    if data == 'Moderate':
        return 1
    if data == 'Considerable -':
        return 2
    if data == 'Considerable +':
        return 3
    if data == 'High':
        return 4

def create_dataset(dataset, look_back=1, label_index=None):
    x, y = [], []
    for index, value in enumerate(dataset):
        try:
            current_index = index + 1
            dataframe_index = current_index + 1
            look_back_index = dataframe_index + look_back
            if look_back_index > len(dataset):
                raise IndexError
            previous = dataset[dataframe_index:look_back_index]
            prediction = dataset[current_index][label_index]
            x.append(previous)
            y.append(prediction)
        except IndexError:
            pass

    return np.array(x), np.array(y)

# load CSV data with specific feature columns
dataset = pd.read_csv(file_name, index_col=False, usecols=features, skipinitialspace=True)
label_index = dataset.columns.get_loc(observed_hazard)

# backfill missing values with earlier values
dataset = dataset.fillna(method='bfill')

# drop remaining un-backfillable rows
dataset = dataset.dropna(subset=features)

# numerical risk values
dataset[observed_hazard] = dataset[observed_hazard].apply(numerical_labels)

# split train / test data
train, test = train_test_split(dataset, test_size=0., shuffle=False)

# create time seriesed dataset and reshape
x_train, y_train = create_dataset(train.values, look_back=look_back,
        label_index=label_index)
x_train = x_train.reshape(x_train.shape[0], 1, look_back*feature_count)

# x_test, y_test = create_dataset(test, look_back=look_back,
#         label_index=label_index)
# x_test = x_test.reshape(x_test.shape[0], 1, look_back*feature_count)

# specify model and compile
model = Sequential()
model.add(LSTM(512, activation='relu', return_sequences=True, input_shape=x_train[0].shape))
model.add(LSTM(512, activation='tanh'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# fit model to dataset
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=None)

# make predictions
train_predict = model.predict(x_train)
# test_predict = model.predict(x_test)

plt.plot(y_train, label='y_train')
plt.plot(train_predict, label='train_predict')
# plt.plot(y_test, label='y_test')
# plt.plot(test_predict, label='test_predict')
plt.legend()
plt.show()