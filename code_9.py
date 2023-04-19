import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# 為替レートのデータを取得する
df = yf.download('USDJPY=X', start='2020-01-01', end='2023-04-18', interval='1d')

# 移動平均線を計算する
ma_5 = df['Close'].rolling(window=5).mean()
ma_10 = df['Close'].rolling(window=10).mean()
ma_20 = df['Close'].rolling(window=20).mean()
ma_60 = df['Close'].rolling(window=60).mean()

# 特徴量として移動平均線を追加する
df['MA_5'] = ma_5
df['MA_10'] = ma_10
df['MA_20'] = ma_20
df['MA_60'] = ma_60

# 欠損値を削除する
df = df.dropna()

# 特徴量として使用する列を選択する
features = ['Close', 'MA_5', 'MA_10', 'MA_20', 'MA_60']
data = df[features].values

# データの正規化を行う
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# 学習データとテストデータに分割する
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# 特徴量とターゲットに分割する
train_X, train_y = train_data[:, 1:], train_data[:, 0]
test_X, test_y = test_data[:, 1:], test_data[:, 0]

# LSTMモデルを作成する
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# モデルをコンパイルする
model.compile(optimizer='adam', loss='mean_squared_error')

# モデルを学習する
model.fit(train_X.reshape(train_X.shape[0], train_X.shape[1], 1), train_y, epochs=50, batch_size=32)

# テストデータを予測する
predicted_y = model.predict(test_X.reshape((test_X.shape[0], test_X.shape[1], 1)))
predicted_y = predicted_y.reshape((predicted_y.shape[0],))

# 予測結果を逆正規化する
predicted_y = scaler.inverse_transform(predicted_y.reshape((-1, 1)))
test_y = scaler.inverse_transform(test_y.reshape((-1, 1)))
predicted_y = predicted_y.reshape((-1,))
test_y = test_y.reshape((-1,))

#グラフを表示する
plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size:], test_y, color='blue', label='actual')
plt.plot(df.index[train_size:], predicted_y, color='red', label='predicted')
plt.plot(df.index[:train_size], df['Close'][:train_size], color='gray', label='train')
plt.legend()
plt.show()
