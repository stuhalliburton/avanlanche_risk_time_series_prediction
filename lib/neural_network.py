from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D

class NeuralNetwork:
    def __init__(self, input_shape):
        model = Sequential()
        # model.add(LSTM(32, activation='tanh', input_shape=input_shape))
        model.add(LSTM(32, activation='tanh', return_sequences=True,
            input_shape=input_shape))
        model.add(Conv1D(32, 1, activation='relu'))
        model.add(LSTM(16, activation='tanh', return_sequences=True))
        model.add(Conv1D(16, 1, activation='relu'))
        model.add(LSTM(8, activation='tanh'))
        model.add(Dense(1, activation='relu'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def train(self, x_train, y_train, epochs=10, batch_size=32, validation_split=0.05):
        return self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
            validation_split=validation_split)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)
