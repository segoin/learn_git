import time
import pyupbit
import datetime
import requests
import numpy as np
import pandas as pd
import pyupbit
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import math 

def MinMaxScaler(data):
    nu  =   data - np.min(data,0)
    de  =   np.max(data,0) - np.min(data,0)
    return nu / (de+1e-7)

def Light_GBM(ticker):
    # OHLCV(open, high, low, close, volume)로 당일 시가,고가,저가,종가,거래량에 대한 데이터
    df = pyupbit.get_ohlcv(ticker, interval="minutes30", count=10000)

