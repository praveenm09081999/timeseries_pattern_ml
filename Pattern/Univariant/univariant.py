import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load your training dataset
df_train = pd.read_csv('./../../Dataset/A&B-pattern2-multivariant.csv')

# Extract features (dt and A)
X_train = df_train[['dt', 'A']].values

# Normalize the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create sequences for the LSTM model
sequence_length = 10
X_train_sequences = []
y_train_labels = []

for i in range(len(X_train_scaled) - sequence_length):
    X_train_sequences.append(X_train_scaled[i:i + sequence_length])
    y_train_labels.append(X_train_scaled[i + sequence_length][1])

X_train_sequences = np.array(X_train_sequences)
y_train_labels = np.array(y_train_labels)

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train_sequences.shape[1], X_train_sequences.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train_sequences, y_train_labels, epochs=50, batch_size=32)

# Save the trained model if needed
model.save('trained_lstm_model.h5')