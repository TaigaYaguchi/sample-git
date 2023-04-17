import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# データの取得
symbol = "JPY=X"  # 日本円/米ドルの為替レート
start_date = "2022-01-01"
end_date = "2022-01-15"
df = yf.download(symbol, start=start_date, end=end_date, interval="1d")
df = df[['Close']]  # 終値のみを使用

# データの前処理
scaler = MinMaxScaler()
data = scaler.fit_transform(df.values)
train_size = int(len(data) * 0.8)  # 80%を訓練データとする
train_data = data[:train_size]
test_data = data[train_size:]
window_size = 10

# 特徴量の作成
def create_features(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

X_train, y_train = create_features(train_data, window_size)
X_test, y_test = create_features(test_data, window_size)

# LSTMモデルの構築
model = Sequential()
model.add(LSTM(units=50, activation="relu", input_shape=(window_size, 1)))
model.add(Dense(units=1))
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=100, batch_size=32)

# 予測
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

# 予測結果と実際の為替レートの表示
plt.plot(df.index[-len(y_test):], y_test, label="Actual")
plt.plot(df.index[-len(y_test):], y_pred, label="Predicted")
plt.xlabel("Date")
plt.ylabel("Exchange Rate")
plt.legend()
plt.show()
