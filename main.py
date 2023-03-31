import ANN_Stocks_SLV as ANNSLV
import ANN_Stocks_APPL as ANNAPPL
import time
import numpy as np

while True:
    stock = input("Enter a stock ticker from the list SLV, APPL:")
    if stock == 'SLV':
        ANNSLV.Start()
    elif stock == 'APPL':
        ANNAPPL.Start()
