import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from feedFromCsv import feedFromCsv
import matplotlib.pyplot as plt

# normalize data
def processData(data, timePortion):
    trainX = []
    trainY = []
    size = len(data)

    features = data.loc[:,'Close'].to_numpy()
    features = features.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(features)
    scaledFeatures = scaler.transform(features).reshape(-1)

    for i in range(timePortion, size):
        for j in range(i - timePortion, i):
            trainX.append(scaledFeatures[j])
        trainY.append(scaledFeatures[i])

    return np.asarray(trainX), np.asarray(trainY)

# auxiliary function
def generateNextDayPrediction(data, timePortion):
    size = len(data)
    features = []

    for i in range(size - timePortion, size):
        features.append(data[i])

    return np.asarray(features)

# number of days in a batch
timePortion = 7
# import stock data
df = feedFromCsv('cdr')
originalData = df.loc[:,'Close'].to_numpy()
# form train and validation data
trainX, trainY = processData(df[:900], timePortion)
testX, testY = processData(df[900:], timePortion)




# build neural network
model = models.Sequential()
model.add(layers.Conv1D(128, 2, activation='relu', input_shape=(7, 1)))
model.add(layers.AveragePooling1D(2, 1))
model.add(layers.Conv1D(64, 2, activation='relu'))
model.add(layers.AveragePooling1D(2, 1))
model.add(layers.Flatten())
model.add(layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
# train and validate
history = model.fit(trainX.reshape(-1, timePortion, 1), trainY, epochs=100, validation_data=(testX.reshape(-1, timePortion, 1), testY))

# predictedX = model.predict(trainX.reshape(-1, timePortion, 1))

# Predict stock for next day
# nextDayPrediction = generateNextDayPrediction(originalData, timePortion)
# scaler = MinMaxScaler()
# scaler.fit(nextDayPrediction.reshape(-1, 1))
# nextDayPredictionScaled = scaler.transform(nextDayPrediction.reshape(-1, 1)).reshape(-1)
# tensorNextDayPrediction = nextDayPredictionScaled.reshape(1, timePortion, 1)
# predictedValue = model.predict(tensorNextDayPrediction)

# inversePredictedValue = scaler.inverse_transform(predictedValue)

# print(f'predicted value for next day: {inversePredictedValue}')

# Plot learning rate
# plt.plot(history.history['mean_squared_error'], "b--", label="MSE dla danych uczących")
# plt.plot(history.history['mean_absolute_error'], "g--", label="MAE of training data")
# plt.plot(history.history['val_mean_squared_error'], "b", label="MSE dla danych walidacyjnych")
# plt.plot(history.history['val_mean_absolute_error'], "g", label="MAE of validation")
# plt.ylabel('MSE')
# plt.xlabel('Iteracja')
# plt.ylim(0)
# plt.legend()
# plt.show()
