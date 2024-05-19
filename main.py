import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Load the data
file_path = 'sample_data_set.xlsx'
df = pd.read_excel(file_path)

# Handle missing values by forward filling
df.ffill(inplace=True)

# Function to prepare the data for LSTM
def create_dataset(data, time_step=3):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Define model predict function with @tf.function outside the loop
@tf.function
def model_predict(model, input_data):
    return model(input_data)

# Function to forecast future values using the LSTM model
def forecast_future(data, model, time_step=3, n_future=3):
    future_predictions = []
    last_sequence = data[-time_step:]  # Get the last sequence
    current_input = last_sequence.reshape((1, time_step, 1))

    for _ in range(n_future):
        next_pred = model_predict(model, current_input)[0]
        future_predictions.append(next_pred)
        current_input = np.append(current_input[:, 1:, :], [[next_pred]], axis=1)

    return np.array(future_predictions)

# Function to calculate Mean Absolute Percentage Error (MAPE) with handling for zero actual values
def calculate_mape(actual, forecast, epsilon=1e-10):
    actual, forecast = np.array(actual), np.array(forecast)
    actual = np.where(actual == 0, epsilon, actual)  # Replace zero actual values with epsilon
    return np.mean(np.abs((actual - forecast) / actual)) * 100

# Function to prepare the data for LSTM for the first 15 rows of the dataset
def prepare_lstm_data(df, time_step=3, rows=15):
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
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        # Train the model
        model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=0, callbacks=[early_stopping])
        
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

# Function to forecast using ARIMA model
def arima_forecasting(data, start_year_column, rows=15):
    pred_2020, pred_2021, pred_2022 = [], [], []
    mape_2020, mape_2021, mape_2022 = [], [], []

    for index, row in data.iloc[:rows].iterrows():
        ts = row.iloc[start_year_column:row.index.get_loc('2019') + 1].values.astype(float)
        
        model = ARIMA(ts, order=(1, 1, 1))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=3)
        
        pred_2020.append(forecast[0])
        pred_2021.append(forecast[1])
        pred_2022.append(forecast[2])
        
        mape_2020.append(mean_absolute_percentage_error([row['2020']], [forecast[0]]))
        mape_2021.append(mean_absolute_percentage_error([row['2021']], [forecast[1]]))
        mape_2022.append(mean_absolute_percentage_error([row['2022']], [forecast[2]]))

    return pred_2020, pred_2021, pred_2022, mape_2020, mape_2021, mape_2022

# Function to forecast using Prophet model
def prophet_forecasting(data, start_year_column, rows=15):
    pred_2020, pred_2021, pred_2022 = [], [], []
    mape_2020, mape_2021, mape_2022 = [], [], []

    for index, row in data.iloc[:rows].iterrows():
        ts = row.iloc[start_year_column:row.index.get_loc('2019') + 1].astype(float)
        ts.index = pd.to_datetime(ts.index, format='%Y')

        df = pd.DataFrame({'ds': ts.index, 'y': ts.values})

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=3, freq='Y')
        forecast = model.predict(future)

        # Extract the forecasts for 2020, 2021, and 2022
        forecast_2020 = forecast.loc[forecast['ds'] == '2020-12-31', 'yhat'].values
        forecast_2021 = forecast.loc[forecast['ds'] == '2021-12-31', 'yhat'].values
        forecast_2022 = forecast.loc[forecast['ds'] == '2022-12-31', 'yhat'].values

        # Append forecasts
        pred_2020.append(forecast_2020[0] if len(forecast_2020) > 0 else None)
        pred_2021.append(forecast_2021[0] if len(forecast_2021) > 0 else None)
        pred_2022.append(forecast_2022[0] if len(forecast_2022) > 0 else None)
        
        # Calculate MAPE if forecast is available
        mape_2020.append(mean_absolute_percentage_error([row['2020']], [forecast_2020[0]]) if len(forecast_2020) > 0 else None)
        mape_2021.append(mean_absolute_percentage_error([row['2021']], [forecast_2021[0]]) if len(forecast_2021) > 0 else None)
        mape_2022.append(mean_absolute_percentage_error([row['2022']], [forecast_2022[0]]) if len(forecast_2022) > 0 else None)

    return pred_2020, pred_2021, pred_2022, mape_2020, mape_2021, mape_2022

# Start year column
start_year_column = int(input("Please enter the column number where the time series starts: "))

# Apply ARIMA model
arima_pred_2020, arima_pred_2021, arima_pred_2022, arima_mape_2020, arima_mape_2021, arima_mape_2022 = arima_forecasting(df, start_year_column, rows=15)

# Apply Prophet model
prophet_pred_2020, prophet_pred_2021, prophet_pred_2022, prophet_mape_2020, prophet_mape_2021, prophet_mape_2022 = prophet_forecasting(df, start_year_column, rows=15)

# Prepare LSTM data
lstm_predictions, mape_scores = prepare_lstm_data(df, rows=15)

# Update the dataframe with the predictions and MAPE values
for i in range(15):
    df.loc[i, '2020_arima_pred'] = arima_pred_2020[i]
    df.loc[i, '2021_arima_pred'] = arima_pred_2021[i]
    df.loc[i, '2022_arima_pred'] = arima_pred_2022[i]
    df.loc[i, '2020_arima_mape'] = arima_mape_2020[i]
    df.loc[i, '2021_arima_mape'] = arima_mape_2021[i]
    df.loc[i, '2022_arima_mape'] = arima_mape_2022[i]
    
    df.loc[i, '2020_prophet_pred'] = prophet_pred_2020[i]
    df.loc[i, '2021_prophet_pred'] = prophet_pred_2021[i]
    df.loc[i, '2022_prophet_pred'] = prophet_pred_2022[i]
    df.loc[i, '2020_prophet_mape'] = prophet_mape_2020[i]
    df.loc[i, '2021_prophet_mape'] = prophet_mape_2021[i]
    df.loc[i, '2022_prophet_mape'] = prophet_mape_2022[i]

    df.loc[i, '2020_lstm_pred'] = lstm_predictions[i][0]
    df.loc[i, '2020_lstm_mape'] = mape_scores[i][0]
    df.loc[i, '2021_lstm_pred'] = lstm_predictions[i][1]
    df.loc[i, '2021_lstm_mape'] = mape_scores[i][1]
    df.loc[i, '2022_lstm_pred'] = lstm_predictions[i][2]
    df.loc[i, '2022_lstm_mape'] = mape_scores[i][2]

# Save the updated dataframe to a new Excel file
output_file_path = 'updated_forecasted_data_set_combined.xlsx'
df.to_excel(output_file_path, index=False)
