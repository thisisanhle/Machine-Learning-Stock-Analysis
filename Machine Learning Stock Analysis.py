# Importing Libraries
import requests
import os
import xlsxwriter
import math
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
#from bs4 import BeautifulSoup
#from selenium import webdriver
#from selenium.webdriver.chrome.options import Options

# Stock Prediction
def stockPrediction(stockName, days):
    print(stockName)
    print(days)
    # Load Data
    company=stockName

    start = dt.datetime(2018, 1, 1)
    end = dt.datetime(2021, 1, 1)
    
    data = web.DataReader(company, 'yahoo', start, end)

    # Prepare Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledData = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    predictionDays = 60

    xTrain=[]
    yTrain=[]

    for x in range(predictionDays, len(scaledData)):
        xTrain.append(scaledData[x-predictionDays:x, 0])
        yTrain.append(scaledData[x, 0])

    xTrain, yTrain = np.array(xTrain), np.array(yTrain)
    xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

    # Build the model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(xTrain.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    # Prediction of the next closing value
    model.add(Dense(units=days))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train Data
    model.fit(xTrain, yTrain, epochs=25, batch_size=32)

    # Test the model accuracy on existing data
    testStart = dt.datetime(2021, 1, 1)
    testEnd = dt.datetime.now()

    testData = web.DataReader(company, 'yahoo', testStart, testEnd)
    actualPrices = testData['Close'].values
    
    totalDataSet = pd.concat((data['Close'], testData['Close']), axis=0)

    modelInputs = totalDataSet[len(totalDataSet)-len(testData)-predictionDays:].values
    modelInputs = modelInputs.reshape(-1, 1)
    modelInputs = scaler.transform(modelInputs)

    # Make predictions on test data
    xTest=[]

    for x in range(predictionDays, len(modelInputs)):
        xTest.append(modelInputs[x-predictionDays:x, 0])

    xTest=np.array(xTest)
    xTest=np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))    

    predictedPrices=model.predict(xTest)
    predictedPrices=scaler.inverse_transform(predictedPrices)

    # Plot the test predictions
    plt.plot(actualPrices, color="black", label=f"Actual {company} price")
    plt.plot(predictedPrices, color="green", label=f"Predicted {company} price")
    plt.title(f"{company} share price")
    plt.xlabel("Time")
    plt.ylabel(f"{company} share price")
    plt.legend()
    plt.show()
    
    # Predict next day
    realData = [modelInputs[len(modelInputs)+1-predictionDays : len(modelInputs+1), 0]]
    realData = np.array(realData)
    realData = np.reshape(realData, (realData.shape[0], realData.shape[1], 1))

    prediction = model.predict(realData)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction: {prediction}")
    return prediction

# Buy Stock
def buyStock(stockName):
    # Load Data
    company=stockName

    #List with average minimum for week, month, and 3 months
    averages = []
    for i in range(3):
        # Calculate the average minimum for the past week
        if i==0:
            start = dt.datetime.now() - dt.timedelta(days=7)
            end = dt.datetime.now()
            
            data = web.DataReader(company, 'yahoo', start, end)

            dates =[]
            for x in range(len(data)):
                newdate = str(data.index[x])
                newdate = newdate[0:10]
                dates.append(newdate)

            data['dates'] = dates

            averages.append(data['Low'].mean())

        # Calculate the average minimum for the past month
        if i==1:
            start = dt.datetime.now() - dt.timedelta(days=30)
            end = dt.datetime.now()
            
            data = web.DataReader(company, 'yahoo', start, end)

            dates =[]
            for x in range(len(data)):
                newdate = str(data.index[x])
                newdate = newdate[0:10]
                dates.append(newdate)

            data['dates'] = dates

            averages.append(data['Low'].mean())

        # Calculate the average minimum for the past 3 months
        if i==2:
            start = dt.datetime.now() - dt.timedelta(days=90)

            end = dt.datetime.now()
            
            data = web.DataReader(company, 'yahoo', start, end)

            dates =[]
            for x in range(len(data)):
                newdate = str(data.index[x])
                newdate = newdate[0:10]
                dates.append(newdate)

            data['dates'] = dates

            averages.append(data['Low'].mean())

    #print(averages)
    return averages

# Sell Stock
def sellStock(stockName):
    # Load Data
    company=stockName

    #List with average maximums for week, month, and 3 months
    averages = []
    for i in range(3):
        # Calculate the average maximums for the past week
        if i==0:
            start = dt.datetime.now() - dt.timedelta(days=7)
            end = dt.datetime.now()
            
            data = web.DataReader(company, 'yahoo', start, end)

            dates =[]
            for x in range(len(data)):
                newdate = str(data.index[x])
                newdate = newdate[0:10]
                dates.append(newdate)

            data['dates'] = dates

            averages.append(data['High'].mean())

        # Calculate the average maximums for the past month
        if i==1:
            start = dt.datetime.now() - dt.timedelta(days=30)
            end = dt.datetime.now()
            
            data = web.DataReader(company, 'yahoo', start, end)

            dates =[]
            for x in range(len(data)):
                newdate = str(data.index[x])
                newdate = newdate[0:10]
                dates.append(newdate)

            data['dates'] = dates

            averages.append(data['High'].mean())

        # Calculate the average maximums for the past 3 months
        if i==2:
            start = dt.datetime.now() - dt.timedelta(days=90)

            end = dt.datetime.now()
            
            data = web.DataReader(company, 'yahoo', start, end)

            dates =[]
            for x in range(len(data)):
                newdate = str(data.index[x])
                newdate = newdate[0:10]
                dates.append(newdate)

            data['dates'] = dates

            averages.append(data['High'].mean())

    #print(averages)
    return averages
    
def setPrices(minAverages, maxAverages):
    averageBuy=[]
    averageSell=[]
    minAveragesModified=[]
    maxAveragesModified=[]
    for i in range(3):
        minAveragesModified.append(minAverages[i]+(maxAverages[i]-minAverages[i])*0.1)
        maxAveragesModified.append(maxAverages[i]-(maxAverages[i]-minAverages[i])*0.1)
    #print(minAveragesModified)
    #print(maxAveragesModified)
    for i in range(3):
        if i==0:
            averageBuy.append(minAveragesModified[i]*0.45)
            averageSell.append(maxAveragesModified[i]*0.45)
        if i==1:
            averageBuy.append(minAveragesModified[i]*0.35)
            averageSell.append(maxAveragesModified[i]*0.35)
        if i==2:
            averageBuy.append(minAveragesModified[i]*0.20)
            averageSell.append(maxAveragesModified[i]*0.20)

    #print(averageBuy)
    #print(averageSell)

    buyAt=np.sum(averageBuy)
    sellAt=np.sum(averageSell)

    print('Buy at: ', buyAt)
    print('Sell at: ', sellAt)

def stockStrength(stockName):
    # Load Data
    company=stockName
    
    start = dt.datetime(2020, 5, 1)
    end = dt.datetime.now()
    
    data = web.DataReader(company, 'yahoo', start, end)

    # Prepare data to calculate RSI
    # Get the difference in price from the previous day
    delta = data['Adj Close'].diff(1)

    # Remove the NaN from delta
    delta = delta.dropna()

    # Get the positive gains
    up = delta.copy()
    # Get the negative gains
    down = delta.copy()

    # Get only the positive values in up and negative values in down
    up[up<0]=0
    down[down>0]=0

    # Set the time period
    period = 14

    # Calculate the average gain and loss
    avgGain = up.rolling(window=period).mean()
    avgLoss = abs(down.rolling(window=period).mean())

    # Calculate the relative strength (RS)
    rs = avgGain/avgLoss
    # Calculate the relative strength index (RSI)
    rsi = 100.0-(100.0/(1.0+rs))

    # Show the rsi visually
    # Create a new data
    newData = pd.DataFrame()
    newData['Adjusted Close Price'] = data['Adj Close']
    newData['RSI'] = rsi

    # Plot the adjusted close price 
    plt.figure(figsize=(12.2, 4.5))
    plt.plot(newData.index, newData['Adjusted Close Price'])
    plt.title('Adjusted Close Price History')
    plt.legend(newData.columns.values, loc = 'upper left')
    #plt.show()
    # Plot the RSI with significant levels
    plt.figure(figsize=(12.2, 4.5))
    plt.title('RSI Plot')
    plt.plot(newData.index, newData['RSI'])

    plt.axhline(0, linestyle='--', alpha=0.5, color='gray')
    plt.axhline(10, linestyle='--', alpha=0.5, color='orange')
    plt.axhline(20, linestyle='--', alpha=0.5, color='green')
    plt.axhline(30, linestyle='--', alpha=0.5, color='red')
    plt.axhline(70, linestyle='--', alpha=0.5, color='red')
    plt.axhline(80, linestyle='--', alpha=0.5, color='green')
    plt.axhline(90, linestyle='--', alpha=0.5, color='orange')
    plt.axhline(100, linestyle='--', alpha=0.5, color='gray')

    plt.show()

def stockDecider(stockName):
    #buyStock(stockName)
    #sellStock(stockName)
    setPrices(buyStock(stockName), sellStock(stockName))
    stockStrength(stockName)
    stockPrediction(stockName, 1)

def main():
    stockDecider('mrna')
    
main()