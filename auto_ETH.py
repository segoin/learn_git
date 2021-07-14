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

access = "ccyUip4ujykayb5qFP9xkncrbbt9HPM98vtVQeqV"
secret = "x3fUq5n4MMHDyKDKeNZdVOcoMjiWQqB0pBBErDjn"
myToken = "xoxb-1702501177444-1689568357894-eTntzP8uGJlB0sDjbpfDP5mK"

def MinMaxScaler(data):
    nu  =   data - np.min(data,0)
    de  =   np.max(data,0) - np.min(data,0)
    return nu / (de+1e-7)

def Light_GBM(ticker):
    # OHLCV(open, high, low, close, volume)로 당일 시가,고가,저가,종가,거래량에 대한 데이터
    df = pyupbit.get_ohlcv(ticker, interval="minutes30", count=10000)

    # 익일종가
    df['close_nextday'] = df['close'].shift(-1)
    df['volume_mean_3day'] = df['volume'].rolling(window=3).mean()
    df['vloume_inc_yn_vs_last_1day'] = np.where(df['volume'] > df['volume_last_1day'],1,0)
    df['bol_upper'] = df['MA20'] + ( 2 * df['close'].rolling(window=20).std() )

