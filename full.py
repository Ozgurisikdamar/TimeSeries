import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_percentage_error
import warnings

warnings.filterwarnings("ignore")

# Veri setini yükle
df = pd.read_excel('sample_data_set.xlsx')

# İlk 15 satırı seç
df = df.head(15)

# Eksik değerleri doldur (örneğin, ileri doldurma yöntemiyle)
df.fillna(method='ffill', inplace=True)

# Sütun adlarını kontrol et
print(df.columns)

# Zaman sütunu olduğunu varsaydığımız sütunu datetime formatına çevirelim
# Burada 'time' veya 'date' sütunu olabilir. Kendi sütun adınıza göre değiştirin.
time_column = 'time'  # 'date' yerine geçecek sütun adını buraya yazın
df[time_column] = pd.to_datetime(df[time_column])

# ARIMA modeli ile tahmin
def arima_forecast(train, order, steps):
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Prophet modeli ile tahmin
def prophet_forecast(train, steps):
    model = Prophet()
    train_df = pd.DataFrame({'ds': train.index, 'y': train.values})
    model.fit(train_df)
    future = model.make_future_dataframe(periods=steps, freq='D')
    forecast = model.predict(future)
    return forecast['yhat'].iloc[-steps:]

# LSTM modeli ile tahmin
def lstm_forecast(train, steps, n_features=1):
    train = np.array(train)
    train = train.reshape((train.shape[0], 1, n_features))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(train, train, epochs=300, verbose=0)

    forecast = []
    input_data = train[-1:]
    for _ in range(steps):
        yhat = model.predict(input_data, verbose=0)
        forecast.append(yhat[0][0])
        input_data = np.append(input_data[:, 1:], [[yhat]], axis=1)
    return forecast

# Örnek veri ile çalışmak için ilk 10 satırı alalım
train_data = df['value'][:10]

# ARIMA tahmini ve MAPE
arima_pred = arima_forecast(train_data, order=(5,1,0), steps=5)
arima_mape = mean_absolute_percentage_error(df['value'][10:15], arima_pred)

# Prophet tahmini ve MAPE
prophet_pred = prophet_forecast(train_data, steps=5)
prophet_mape = mean_absolute_percentage_error(df['value'][10:15], prophet_pred)

# LSTM tahmini ve MAPE
lstm_pred = lstm_forecast(train_data, steps=5)
lstm_mape = mean_absolute_percentage_error(df['value'][10:15], lstm_pred)

# Tahmin ve MAPE değerlerini dataframe'e ekleyelim
df['arima_2020_pred'] = np.nan
df['arima_2020_mape'] = np.nan
df['prophet_2020_pred'] = np.nan
df['prophet_2020_mape'] = np.nan
df['lstm_2020_pred'] = np.nan
df['lstm_2020_mape'] = np.nan

df.loc[10:14, 'arima_2020_pred'] = arima_pred.values
df.loc[10:14, 'arima_2020_mape'] = arima_mape
df.loc[10:14, 'prophet_2020_pred'] = prophet_pred.values
df.loc[10:14, 'prophet_2020_mape'] = prophet_mape
df.loc[10:14, 'lstm_2020_pred'] = lstm_pred
df.loc[10:14, 'lstm_2020_mape'] = lstm_mape

# Sonuçları gözlemleyelim
print(df)

# Excel dosyasına yazalım
df.to_excel('forecast_results.xlsx', index=False)
