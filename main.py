import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lib.data_formatter import DataFormatter
from lib.neural_network import NeuralNetwork

# create time seriesed dataset
X, Y = DataFormatter().create_dataset(look_back=7)

# split train / test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, shuffle=False)

# build neural network
network = NeuralNetwork(input_shape=x_train[0].shape)

# train network
training = network.train(x_train, y_train, epochs=300, batch_size=32, validation_split=0.05)

# evaluate model against test dataset
scores = network.evaluate(x_test, y_test)
print 'Test Loss: {}, Test Accuracy {}'.format(scores[0], scores[1])

# make predictions
train_predict = network.predict(x_train)
test_predict = network.predict(x_test)

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
