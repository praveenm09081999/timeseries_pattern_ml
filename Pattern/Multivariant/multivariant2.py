import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def predictModel2(df_test):
    model = load_model('Pattern/Multivariant/trained_lstm_model_2.h5')
    df_train = pd.read_csv('Dataset/A&B-pattern2-multivariant.csv')

    X_train = df_train[['dt', 'A', 'B']].values

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    sequence_length = 10

    X_test = df_test[['dt', 'A', 'B']].values

    X_test_scaled = scaler.transform(X_test)

    X_test_sequences = []

    for i in range(len(X_test_scaled) - sequence_length):
        X_test_sequences.append(X_test_scaled[i:i + sequence_length])

    X_test_sequences = np.array(X_test_sequences)

    predicted_labels = model.predict(X_test_sequences)
    print(predicted_labels)

    threshold = 1.2

    occurrences = []
    current_occurrence = []

    for i, value in enumerate(predicted_labels):
        if value > threshold:
            current_occurrence.append(i)
        else:
            if current_occurrence:
                occurrences.append(current_occurrence)
                current_occurrence = []
    return (f"Occurrences of pattern 2 in the test data: {len(occurrences)}")
