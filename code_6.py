import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# データの取得
start_date = '2023-01-01'
end_date = '2023-04-16'
ticker = 'JPY=X'  # 円ドルの為替レートを表すシンボル
data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
data = data[['Close']]
data = data.dropna()
data = data.resample('D').ffill()  # 欠損値を前日の値で補完
y = data.values

# データの前処理
scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler.fit_transform(y)

# 訓練データとテストデータに分割
train_size = int(len(y_scaled) * 0.8)
test_size = len(y_scaled) - train_size
train, test = y_scaled[0:train_size,:], y_scaled[train_size:len(y_scaled),:]

# LSTMモデルの作成と予測
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

lstm_model = create_lstm_model()
lstm_model.fit(np.array(train).reshape(-1, 1, 1), np.array(train), epochs=50, batch_size=1, verbose=2)
lstm_forecast = lstm_model.predict(np.array(test).reshape(-1, 1, 1))
lstm_forecast = scaler.inverse_transform(lstm_forecast)

# グラフで予測結果を表示
plt.figure(figsize=(12, 6))
plt.plot(data. index[-test_size:], y[train_size:len(y_scaled),:], label='Actual')
plt.plot(data.index[-test_size:], lstm_forecast, label='LSTM Forecast')
plt.xlabel('Date')
plt.ylabel('Exchange Rate (JPY/USD)')
plt.title('Exchange Rate (JPY/USD) Forecast')
plt.legend()
plt.show()
