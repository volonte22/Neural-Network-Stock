import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
import keras as keras
import time
import datetime
from csv import writer
import csv



# run and get model
def Start():
    # CREATE PREDICTION MODEL
    dataset_train = pd.read_csv('SLV ALLTIME.csv')  # read csv
    training_set = dataset_train.iloc[:, 1:2].values

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X = input("Make a new model? T/F: ")
    count = 0
    while (count == 0):
        if X == 'T':
            count = 1
        elif X == 'F':
            count = 1
        elif X != 'T':
            X = input("Make a new model? T/F: ")

    if (X == 'T'):
        # CREATE PREDICTION LMSA MODEL

        X_train = []
        y_train = []
        for i in range(60, 4092):
            X_train.append(training_set_scaled[i - 60:i, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=100, batch_size=32)

        # save prediction model
        model.save('SLVALLTIMEMODEL')
    else:
        model = keras.models.load_model('SLVALLTIMEMODEL')

    ####################################################################################
    X = input("Test on old or new models? 'new or old'")
    if X == 'old':
        # TEST DATA ON TRAINING SET PREDICTION MODELS

        # june to december 2022
        dataset_test = pd.read_csv('SLV_JUNE2022_DEC2022.csv')
        real_stock_price = dataset_test.iloc[:, 1:2].values

        dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
        inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = sc.transform(inputs)
        X_test = []
        for i in range(60, 175):
            X_test.append(inputs[i - 60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        plt.plot(real_stock_price, color='black', label='SLV Stock Price')
        plt.plot(predicted_stock_price, color='green', label='Predicted SLV Stock Price')
        plt.title('SLV Stock Price Prediction - June2022 -> Dec2022')
        plt.xlabel('Time')
        plt.ylabel('SLV Stock Price')
        plt.legend()
        plt.show()

        # oct to dec 2022

        dataset_test = pd.read_csv('SLV_OCT2022_DEC2022.csv')
        real_stock_price2 = dataset_test.iloc[:, 1:2].values

        dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
        inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = sc.transform(inputs)
        X_test = []
        for i in range(60, 90):
            X_test.append(inputs[i - 60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        print(X_test)
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        plt.plot(real_stock_price2, color='black', label='SLV Stock Price')
        plt.plot(predicted_stock_price, color='green', label='Predicted SLV Stock Price')
        plt.title('SLV Stock Price Prediction - OCT2022 -> Dec2022')
        plt.xlabel('Time')
        plt.ylabel('SLV Stock Price')
        plt.legend()
        plt.show()

        ####################################################################################

        # testing future predictions
        i = 0
        fileWrite = open('SLV_NOV30-JAN22.csv')

        dataset_test = pd.read_csv('SLV_OCT2022_DEC2022.csv')
        real_stock_price = dataset_test.iloc[:, 1:2].values
        while i < 50:
            dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
            inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
            inputs = inputs.reshape(-1, 1)
            inputs = sc.transform(inputs)
            X_test = []
            b = 90 + 1
            for i in range(60, b):
                X_test.append(inputs[i - 60:i, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            predicted_stock_price = model.predict(X_test)
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)

            plt.plot(real_stock_price2, color='red', label='SLV Stock Price')
            plt.plot(predicted_stock_price, color='green', label='Predicted SLV Stock Price')
            plt.title('SLV Stock Price Prediction - OCT2022 -> Dec2022')
            plt.xlabel('Time')
            plt.ylabel('SLV Stock Price')
            plt.legend()
            i = i + 1  # next

        plt.show()
    elif X == 'new':
        #get live updated model and give information on that

        ticker = 'SLV'
        period1 = int(time.mktime(datetime.datetime(2022, 12, 1, 23, 59).timetuple()))
        period2 = int(time.mktime(datetime.datetime(2023, 1, 23, 23, 59).timetuple()))
        interval = '1d'  # 1d, 1m

        query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'

        df = pd.read_csv(query_string, index_col=False)
        df.to_csv('SLVCURR.csv')

        dataset_test = df
        real_stock_price2 = dataset_test.iloc[:, 1:2].values

        dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
        inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = sc.transform(inputs)
        X_test = []
        for i in range(60, 94):
            X_test.append(inputs[i - 60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        print(predicted_stock_price)
        plt.plot(real_stock_price2, color='black', label='SLV Stock Price')
        plt.plot(predicted_stock_price, color='green', label='Predicted SLV Stock Price')
        plt.title('SLV Stock Price Prediction - DEC2022 -> Current Date - 2023')
        plt.xlabel('Time - Days')
        plt.ylabel('SLV Stock Price')
        plt.legend()
        plt.show()

    elif X == 'long':

        # predictions into the next week

        # TEST DATA ON TRAINING SET PREDICTION MODELS
        plt.title('SLV Pred OCT-JAN23 -> 7 Days Ahead')


        # oct to jan, predicting into future
        r = 112
        count = 0
        timeThing = 7
        while count <= 90:
            while count <= timeThing:
                dataset_test = pd.read_csv('OCT_JAN23.csv')
                real_stock_price = dataset_test.iloc[:, 1:2].values

                dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
                inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
                inputs = inputs.reshape(-1, 1)
                inputs = sc.transform(inputs)
                X_test = []
                for i in range(60, r):
                    X_test.append(inputs[i - 60:i, 0])
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                predicted_stock_price = model.predict(X_test)
                predicted_stock_price = sc.inverse_transform(predicted_stock_price)

                # write back to csv
                newOpen = predicted_stock_price[len(predicted_stock_price)-1][0]
                row_contents = [0, newOpen, 0,0,0,0,0] # 7 total rows
                with open('OCT_JAN23.csv', 'a', newline='') as write_obj:
                    csv_writer = writer(write_obj)
                    #csv_writer.writerow('\n')
                    csv_writer.writerow(row_contents)
                    write_obj.close()

                r = r + 1
                count = count + 1
                print(count)


            # plotme
            if count > 8:
                plt.plot(real_stock_price, color='green', label='SLV Pred Price')
            else:
                plt.plot(real_stock_price, color='black', label='SLV Stock Price')
                plt.plot(predicted_stock_price, color='green', label='Predicted SLV Stock Price')
            plt.xlabel('Time - Days')
            plt.ylabel('SLV Stock Price')
            plt.legend()
            plt.show()

            if timeThing == 7:
                timeThing = 30
                plt.title('SLV Pred OCT-JAN23 -> 30 Days Ahead')
            elif timeThing == 30:
                timeThing = 90
                plt.title('SLV Pred OCT-JAN23 -> 90 Days Ahead')
