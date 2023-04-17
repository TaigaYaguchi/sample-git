import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# データの取得
symbol = 'JPY=X'  # JPY/USDの為替レート
start_date = '2020-01-01'
end_date = '2020-12-31'
df = yf.download(symbol, start=start_date, end=end_date, interval='1d')

# 特徴量の作成
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA60'] = df['Close'].rolling(window=60).mean()
df['Volatility'] = (df['High'] - df['Low']) / df['Close']
df = df.dropna()

# データの分割
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train_data = df.iloc[:train_size, :]
test_data = df.iloc[train_size:, :]

# データの正規化
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data[['Close', 'MA5', 'MA10', 'MA20', 'MA60', 'Volatility']])
test_scaled = scaler.transform(test_data[['Close', 'MA5', 'MA10', 'MA20', 'MA60', 'Volatility']])

# 特徴量とターゲットの作成
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps), :])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 30
X_train, y_train = create_dataset(train_scaled, train_scaled[:, 0], time_steps)
X_test, y_test = create_dataset(test_scaled, test_scaled[:, 0], time_steps)

# LSTMモデルの構築
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')

# モデルの学習
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# モデルによる予測
y_pred = model.predict(X_test)

# y_predを2次元配列に変換
y_pred = y_pred.reshape(-1, 1)

# y_testの形状に合わせてy_predをスライス
y_pred = y_pred[-len(y_test):]

# y_predをy_testの後ろに連結
y_pred = scaler.inverse_transform(np.concatenate((y_test.values.reshape(-1, 1), y_pred), axis=1))[:, -1]


# 予測結果の表示
df_test = test_data.iloc[time_steps:]
df_test['Predicted_Close'] = y_pred
df_test = df_test.reset_index()

# グラフの表示
plt.figure(figsize=(12, 6))
plt.plot(df_test['Date'], df_test['Close'], label='Actual Close')
plt.plot(df_test['Date'], df_test['Predicted_Close'], label='Predicted Close')
plt.xlabel('Date')
plt.ylabel('JPY/USD Exchange Rate')
plt.title('Actual vs. Predicted JPY/USD Exchange Rate')
plt.legend()
plt.show()