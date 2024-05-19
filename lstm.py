import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# Load the data
file_path = 'forecasted_data_set.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Handle missing values by forward filling
df.fillna(method='ffill', inplace=True)

# Function to prepare the data for LSTM
def create_dataset(data, time_step=3):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Function to forecast future values using the LSTM model
def forecast_future(data, model, time_step=3, n_future=3):
    future_predictions = []
    last_sequence = data[-time_step:]  # Get the last sequence
    current_input = last_sequence.reshape((1, time_step, 1))

    for _ in range(n_future):
        next_pred = model.predict(current_input)[0]
        future_predictions.append(next_pred)
        current_input = np.append(current_input[:, 1:, :], [[next_pred]], axis=1)

    return np.array(future_predictions)

# Function to calculate Mean Absolute Percentage Error (MAPE) with handling for zero actual values
def calculate_mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    mask = actual != 0
    if np.sum(mask) == 0:
        return np.inf  # All actual values are zero
    return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100

# Function to prepare the data for LSTM for the first 30 rows of the dataset
def prepare_lstm_data(df, time_step=3, rows=30):
    lstm_predictions = []
    mape_scores = []

    # Iterate over each row (product/sector)
    for i in range(rows):
        # Extract the time series data
        ts_data = df.iloc[i, 2:19].values.reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler()
        ts_data_scaled = scaler.fit_transform(ts_data)
        
        # Prepare the dataset
        X, Y = create_dataset(ts_data_scaled, time_step)
        
        # Split the data into training and testing sets
        train_size = int(len(X) * 0.8)
        test_size = len(X) - train_size
        X_train, X_test = X[:train_size], X[train_size:]
        Y_train, Y_test = Y[:train_size], Y[train_size:]
        
        # Reshape input to be [samples, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], time_step, 1)
        X_test = X_test.reshape(X_test.shape[0], time_step, 1)
        
        # Build the LSTM model
        model = Sequential()
        model.add(Input(shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model
        model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=0)
        
        # Forecast future values for 2020, 2021, 2022
        future_predictions_scaled = forecast_future(ts_data_scaled, model, time_step, 3)
        future_predictions = scaler.inverse_transform(future_predictions_scaled)
        
        # Extract the actual values for 2020, 2021, 2022
        actual_values = df.iloc[i, 16:19].values
        
        # Calculate MAPE
        mape_2020 = calculate_mape([actual_values[0]], [future_predictions[0][0]])
        mape_2021 = calculate_mape([actual_values[1]], [future_predictions[1][0]])
        mape_2022 = calculate_mape([actual_values[2]], [future_predictions[2][0]])
        
        # Append predictions and MAPE scores
        lstm_predictions.append([future_predictions[0][0], future_predictions[1][0], future_predictions[2][0]])
        mape_scores.append([mape_2020, mape_2021, mape_2022])
    
    return lstm_predictions, mape_scores

# Prepare the LSTM data for the first 30 rows of the dataset
lstm_predictions, mape_scores = prepare_lstm_data(df, rows=30)

# Update the dataframe with the predictions and MAPE values
for i in range(30):
    df.loc[i, '2020_lstm_pred'] = lstm_predictions[i][0]
    df.loc[i, '2020_lstm_mape'] = mape_scores[i][0]
    df.loc[i, '2021_lstm_pred'] = lstm_predictions[i][1]
    df.loc[i, '2021_lstm_mape'] = mape_scores[i][1]
    df.loc[i, '2022_lstm_pred'] = lstm_predictions[i][2]
    df.loc[i, '2022_lstm_mape'] = mape_scores[i][2]

# Save the updated dataframe to a new Excel file
output_file_path = 'updated_forecasted_data_set_lstm.xlsx'
df.to_excel(output_file_path, index=False)
