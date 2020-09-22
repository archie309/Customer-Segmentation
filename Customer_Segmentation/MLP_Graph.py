# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
# load Mall Customers Data set
dataset = numpy.loadtxt("MallCustomerDataset.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:5]
Y = dataset[:,5]
# create model
model = Sequential()
model.add(Dense(32, input_dim=5, activation='relu'))
model.add(Dense(62,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))


