import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# データの取得
start_date = '2022-01-01'
end_date = '2022-03-31'
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

# ARIMAモデルの作成と予測
arima_model = ARIMA(train, order=(5, 1, 0))
arima_model_fit = arima_model.fit()
arima_forecast = arima_model_fit.forecast(steps=test_size)
# ARIMAの予測結果を逆スケーリング
arima_forecast = scaler.inverse_transform(arima_forecast.reshape(-1, 1))
#arima_forecast = scaler.inverse_transform(arima_forecast)

# SARIMAモデルの作成と予測
sarima_model = SARIMAX(train, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
sarima_model_fit = sarima_model.fit()
sarima_forecast = sarima_model_fit.forecast(steps=test_size)
# SARIMAの予測結果を逆スケーリング
sarima_forecast = scaler.inverse_transform(sarima_forecast.reshape(-1, 1))
# 予測結果を2次元の配列に変形
#sarima_forecast = sarima_forecast.reshape(-1, 1)
#sarima_forecast = scaler.inverse_transform(sarima_forecast)

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

# 予測結果を重み付き平均してアンサンブル学習
weights = [0.1, 0.2, 0.7]
ensemble_forecast = (weights[0] * arima_forecast) + (weights[1] * sarima_forecast) + (weights[2] * lstm_forecast)

# グラフで予測結果を表示
plt.figure(figsize=(12, 6))
plt.plot(data. index[-test_size:], y[train_size:len(y_scaled),:], label='Actual')
plt.plot(data.index[-test_size:], arima_forecast, label='ARIMA Forecast')
plt.plot(data.index[-test_size:], sarima_forecast, label='SARIMA Forecast')
plt.plot(data.index[-test_size:], lstm_forecast, label='LSTM Forecast')
plt.plot(data.index[-test_size:], ensemble_forecast, label='Ensemble Forecast')
plt.xlabel('Date')
plt.ylabel('Exchange Rate (JPY/USD)')
plt.title('Exchange Rate (JPY/USD) Forecast')
plt.legend()
plt.show()
