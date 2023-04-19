import yfinance as yf
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# データの取得
start_date = '2022-01-01'
end_date = '2023-04-17'
ticker = 'JPY=X'  # 円ドルの為替レートを表すシンボル
data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
data = data[['Close']]
data = data.dropna()
data = data.resample('D').ffill()  # 欠損値を前日の値で補完
y = data.values

# データの前処理
scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler.fit_transform(y)

# LSTMモデルの作成と予測
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

window_size = 100

lstm_model = create_lstm_model()
lstm_model.fit(np.array(y_scaled).reshape(-1, 1, 1), np.array(y_scaled), epochs=50, batch_size=1, verbose=2)
lstm_forecast = lstm_model.predict(np.array(y_scaled[:window_size]).reshape(-1, 1, 1))
lstm_forecast = scaler.inverse_transform(lstm_forecast)

data_list = []
today = datetime.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

for i in range(len(lstm_forecast)):
    tomorrow = today + datetime.timedelta(days=1)
    data_list.append(tomorrow)
    today = tomorrow

# グラフで予測結果を表示
plt.figure(figsize=(12, 6))
plt.plot(data_list, lstm_forecast, label='LSTM Forecast')
plt.xlabel('Date')
plt.ylabel('Exchange Rate (JPY/USD)')
plt.title('Exchange Rate (JPY/USD) Forecast')
plt.legend()
plt.show()
