import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


df = pd.read_csv('400_datapoints_15vis.csv')
#print(df)

#select variable for training
cols = list(df)[1:17] 
df_for_training = df[cols].astype('float64')
#print(df_for_training)

## LSTM uses sigmoid and Tanh that are very sensitive to magnitude 
## so values need to be normalized...
scaler = MinMaxScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

trainX = []
trainY = []

n_future = 1
n_past = 100

for i in range (n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1: i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

#now lets define model

model = Sequential()
model.add(LSTM(64, activation = 'tanh', input_shape = (trainX.shape[1], trainX.shape[2]), return_sequences=False))
#model.add(LSTM(32, activation ='tanh', return_sequences=False))
#model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary

#fit model
history = model.fit(trainX, trainY, epochs=2000, batch_size=32, validation_split=0.1, verbose=1)
model.save('lstm_working_hope_.h5')

#plots
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()


#prediction input

#let's forecast......if we want it to predict only one value set it to 1
v_outs = 25
v_predicts = model.predict(trainX[-v_outs:])

#rescale back
v_copies = np.repeat(v_predicts, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(v_copies)[:,0]
print(y_pred_future)