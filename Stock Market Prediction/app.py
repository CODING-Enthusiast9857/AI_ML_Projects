import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model('D:\Old data\Git_Repo\Forked Repositories\AI ML Projects\AI_ML_Projects\Stock Market Prediction\Stock Prediction Model.keras')

st.header("Stock Market Predictor")

stock = st.text_input("Enter Stock Symbol", 'GOOG')

start = '2012-01-01'
end = '2024-03-26'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0 : int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80) : int(len(data))])

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler(feature_range = (0, 1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)

data_test_scaler = scaler.fit_transform(data_test)

st.subheader("Price vs Moving Average 50")
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(6, 4))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader("Price vs Moving Average 50 vs Moving Average 100")
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(6, 4))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'c')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader("Price vs Moving Average 100 vs Moving Average 200")
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(6, 4))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'c')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scaler.shape[0]):
    x.append(data_test_scaler[i-100:i])
    y.append(data_test_scaler[i, 0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale

y = y * scale

st.subheader("Original Price vs Predicted Price")
fig4 = plt.figure(figsize=(6, 4))
plt.plot(predict, 'r', label='original Price')
plt.plot(y, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)
