# 必要なライブラリをインポート
import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

# yfinanceを使って為替データを取得
def get_forex_data():
    forex_data = yf.download("USDJPY=X", start="2022-01-01", end="2022-04-16", interval="1d")
    forex_data.reset_index(inplace=True)
    return forex_data

# 特徴量の作成
def create_features(data):
    data['MA7'] = data['Close'].rolling(window=7).mean()
    data['MA14'] = data['Close'].rolling(window=14).mean()
    data['MA21'] = data['Close'].rolling(window=21).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA100'] = data['Close'].rolling(window=100).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['EMA'] = data['Close'].ewm(span=30, adjust=False).mean()
    data['Volatility'] = data['Close'].rolling(window=21).std() / data['Close'].rolling(window=21).mean()
    return data

# ARIMAモデルの予測
def arima_forecast(data):
    model = ARIMA(data['Close'].values, order=(5,1,0))
    model_fit = model.fit(disp=0)
    forecast = model_fit.forecast(steps=5)
    return forecast

# Prophetモデルの予測
def prophet_forecast(data):
    data_prophet = data[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
    model = Prophet()
    model.fit(data_prophet)
    future = model.make_future_dataframe(periods=5)
    forecast = model.predict(future)
    return forecast.tail(5)['yhat'].values

# LSTMモデルの予測
def lstm_forecast(data):
    dataset = data['Close'].values
    dataset = dataset.reshape(-1, 1)
    dataset = dataset.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    train_size = int(len(dataset) * 0.80)
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    forecast = np.concatenate((np.array(data['Close'].tail(1)) + testPredict.flatten())).tolist()
    return forecast

# アンサンブル学習による予測結果の重み付き平均
def ensemble_forecast(arima_forecast, prophet_forecast, lstm_forecast):
    weights = [0.4, 0.3, 0.3] # 各モデルの重み付け
    forecast = np.average([arima_forecast, prophet_forecast, lstm_forecast], axis=0, weights=weights)
    return forecast

# メイン関数
def main():
    # 為替データの取得
    forex_data = get_forex_data()

    # 特徴量の作成
    forex_data = create_features(forex_data)

    # ARIMAモデルによる予測
    arima_forecast = arima_forecast(forex_data)

    # Prophetモデルによる予測
    prophet_forecast = prophet_forecast(forex_data)

    # LSTMモデルによる予測
    lstm_forecast = lstm_forecast(forex_data)

    # アンサンブル学習による予測結果の重み付き平均
    ensemble_result = ensemble_forecast(arima_forecast, prophet_forecast, lstm_forecast)

    print("ARIMA予測結果:", arima_forecast)
    print("Prophet予測結果:", prophet_forecast)
    print("LSTM予測結果:", lstm_forecast)
    print("アンサンブル学習による予測結果:", ensemble_result)

if __name__ == '__main__':
    main()
