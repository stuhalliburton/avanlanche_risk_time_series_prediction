import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, LSTM

file_name = 'profiles/southern-cairngorms.csv'
# file_name = 'profiles/northern-cairngorms.csv'
# file_name = 'profiles/glencoe.csv'
# file_name = 'profiles/torridon.csv'
# file_name = 'profiles/lochaber.csv'
# file_name = 'profiles/creag-meagaidh.csv'

# dataset columns
observed_hazard = 'Observed aval. hazard'
temp_gradient = 'Max Temp Grad'
hardness_gradient = 'Max Hardness Grad'
snow_depth = 'Total Snow Depth'
drift = 'Drift'
foot_pen = 'Foot Pen'
rain_at_900 = 'Rain at 900'
summit_air_temp = 'Summit Air Temp'
summit_wind_speed = 'Summit Wind Speed'
summit_wind_dir = 'Summit Wind Dir'
no_settle = 'No Settle'
insolation = 'Insolation'
snow_temp = 'Snow Temp'

# derived features
summit_wind_dir_n = 'Summit Wind Dir_n'
summit_wind_dir_ne = 'Summit Wind Dir_ne'
summit_wind_dir_e = 'Summit Wind Dir_e'
summit_wind_dir_se = 'Summit Wind Dir_se'
summit_wind_dir_s = 'Summit Wind Dir_s'
summit_wind_dir_sw = 'Summit Wind Dir_sw'
summit_wind_dir_w = 'Summit Wind Dir_w'
summit_wind_dir_nw = 'Summit Wind Dir_nw'

columns = [observed_hazard, temp_gradient, hardness_gradient, snow_depth, drift,
        foot_pen, rain_at_900, summit_air_temp, summit_wind_speed,
        summit_wind_dir, no_settle, insolation, snow_temp]
features = [observed_hazard, temp_gradient, hardness_gradient, snow_depth, drift,
        foot_pen, rain_at_900, summit_air_temp, summit_wind_speed, no_settle,
        insolation, snow_temp, summit_wind_dir_n, summit_wind_dir_ne,
        summit_wind_dir_e, summit_wind_dir_se, summit_wind_dir_s,
        summit_wind_dir_sw, summit_wind_dir_w, summit_wind_dir_nw]
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

def bearing_classification(bearing):
    if 0 <= bearing <= 22.5:
        return 'n'
    if 337.6 <= bearing <= 360:
        return 'n'
    if 22.6 <= bearing <= 67.5:
        return 'ne'
    if 67.6 <= bearing <= 112.5:
        return 'e'
    if 112.6 <= bearing <= 157.5:
        return 'se'
    if 157.6 <= bearing <= 202.5:
        return 's'
    if 202.6 <= bearing <= 247.5:
        return 'sw'
    if 247.6 <= bearing <= 292.5:
        return 'w'
    if 292.6 <= bearing <= 337.5:
        return 'nw'
    return np.nan

def create_dataset(dataset, look_back=1, randomise=False):
    x, y = [], []
    label_index = dataset.columns.get_loc(observed_hazard)
    values = dataset.values

    for index, value in enumerate(values):
        if index < look_back:
            continue

        look_back_index = index - look_back

        previous = values[look_back_index:index]
        prediction = value[label_index]

        x.append(previous)
        y.append(prediction)

    if randomise == True:
        x, y = shuffle(x, y)

    return np.array(x), np.array(y)

# load CSV data with specific feature columns
dataset = pd.read_csv(file_name, index_col=False, usecols=columns, skipinitialspace=True)

# backfill missing values with earlier values
dataset = dataset.fillna(method='bfill')

# drop remaining un-backfillable rows
dataset = dataset.dropna(subset=columns)

# numerical risk values
dataset[observed_hazard] = dataset[observed_hazard].apply(numerical_labels)

# encode bearing values
dataset[summit_wind_dir] = dataset[summit_wind_dir].apply(bearing_classification)
dataset = pd.get_dummies(dataset, columns=[summit_wind_dir])

# reverse dataset
dataset = dataset.iloc[::-1]

# split train / test data
train, test = train_test_split(dataset, test_size=0.1, shuffle=False)

# create time seriesed dataset and reshape
x_train, y_train = create_dataset(train, look_back=look_back, randomise=True)
x_test, y_test = create_dataset(test, look_back=look_back, randomise=False)

# specify model and compile
model = Sequential()
model.add(LSTM(16, activation='tanh', input_shape=x_train[0].shape))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# fit model to dataset
training = model.fit(x_train, y_train, epochs=400, batch_size=32, validation_split=0.1)

# make predictions
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

# plots figures
fig = plt.figure()

# plot training metrics
plt.subplot(221)
plt.plot(training.history['acc'], label='acc')
plt.plot(training.history['val_acc'], label='val_acc')
plt.legend()
plt.subplot(222)
plt.plot(training.history['loss'], label='loss')
plt.plot(training.history['val_loss'], label='val_loss')
plt.legend()

# plot predictions
plt.subplot(223)
plt.plot(y_train, label='y_train')
plt.plot(train_predict, label='train_predict')
plt.legend()

plt.subplot(224)
plt.plot(y_test, label='y_test')
plt.plot(test_predict, label='test_predict')
plt.legend()

plt.show()
