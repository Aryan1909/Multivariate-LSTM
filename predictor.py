import numpy as np
from keras.models import Sequential
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

# Load the saved LSTM model
model = tf.keras.saving.load_model('lstm_for_14.9k_45vis_1000_10.h5')

# Assuming your input value has 16 parameters
df = pd.read_csv('input_to_predictor.csv')
#print(df)

#select variable for training
cols = list(df)[1:47] 
input_value = df[cols].astype('float64')
# Normalize the input value using the same scaler used during training
# Assuming you have the scaler object saved as 'scaler'
scaler = MinMaxScaler()
scaler = scaler.fit(input_value)
input_value = scaler.transform(input_value)


# Perform the prediction
output_value_scaled = tf.keras.Model.predict(input_value)

# Inverse transform the output value to get the actual output
output_value = scaler.inverse_transform(output_value_scaled)

# Print the output value
print(output_value)