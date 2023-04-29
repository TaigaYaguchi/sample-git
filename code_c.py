import numpy as np
from sklearn.metrics import r2_score
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
# %matplotlib inline

# データの取得
ticker = 'JPY=X'  # 円ドルの為替レートを表すシンボル
df = yf.download(ticker, start='2014-01-01', end=datetime.now(), interval='1d')
df.head()

#プロット
""" plt.figure(figsize=(16,6))
plt.title(s_target + ' Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price USD ($)', fontsize=14)
plt.show() """

# Close(終値)のデータ
data = df.filter(['Close'])
dataset = data.values

# データを0〜1の範囲に正規化
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# 全体の80%をトレーニングデータとして扱う 今は90
training_data_len = int(np.ceil( len(dataset) * .95 ))

# どれくらいの期間をもとに予測するか
window_size = 100

train_data = scaled_data[0:int(training_data_len), :]

# train_dataをx_trainとy_trainに分ける
x_train, y_train = [], []
for i in range(window_size, len(train_data)):
    x_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])

# numpy arrayに変換
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#LSTMモデルの実装と学習
model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train, batch_size=32, epochs=100)

#モデルの概要を表示
model.summary()

# テストデータを作成
test_data = scaled_data[training_data_len - window_size: , :]

x_test = []
y_test = dataset[training_data_len:, :]
x_test.append(test_data[:window_size])

# numpy arrayに変換
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# 予測を実行する
predictions = []
for i in range(len(y_test)):    #for i in range(len(y_test)):
    prediction = model.predict(x_test)
    x_test = x_test.tolist()
    x_test[0] = x_test[0][1:]
    prediction = [prediction[0].tolist()]
    x_test[0].append(prediction[0])
    x_test = np.array(x_test)
    predictions.extend(prediction)
predictions = scaler.inverse_transform(predictions)

# 二乗平均平方根誤差（RMSE）: 0に近いほど良い
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(rmse)

# 決定係数(r2) : 1に近いほど良い
r2s = r2_score(y_test, predictions)
print(r2s)

#予測値（茶色）と実際の株価（赤）をグラフで可視化。
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,6))
plt.xlabel('Date')
plt.ylabel('Exchange Rate (JPY/USD)')
plt.title('Exchange Rate (JPY/USD) Forecast')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()