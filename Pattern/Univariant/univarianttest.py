import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def predictUniModel(df_test):
    model = load_model('Pattern/Univariant/trained_lstm_model.h5')
    df_train = pd.read_csv('Dataset/A-univariant.csv')
    X_train = df_train[['dt', 'A']].values

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    sequence_length = 10

    X_test = df_test[['dt', 'A']].values

    X_test_scaled = scaler.transform(X_test)

    X_test_sequences = []

    for i in range(len(X_test_scaled) - sequence_length):
        X_test_sequences.append(X_test_scaled[i:i + sequence_length])

    X_test_sequences = np.array(X_test_sequences)

    predicted_labels = model.predict(X_test_sequences)

    predicted_labels_original_scale = scaler.inverse_transform(np.column_stack((np.zeros_like(predicted_labels), predicted_labels)))[:, 1]

    threshold = 0.5

    occurrences = []
    current_occurrence = []

    for i, value in enumerate(predicted_labels_original_scale):
        if value > threshold:
            current_occurrence.append(i)
        else:
            if current_occurrence:
                occurrences.append(current_occurrence)
                current_occurrence = []

    return "Occurrences in the test data: " + str(len(occurrences))