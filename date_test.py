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

    # 익일종가 활용 lable 생성
    df['close_nextday'] = df['close'].shift(-1)
    df['high_nextday'] = df['high'].shift(-1)
    df['low_nextday'] = df['low'].shift(-1)
    df['up_yn'] = np.where( (df['high_nextday'] - df['close'])/df['close'] > 0.005,1,0)
    df['down_yn'] = np.where( (df['close'] - df['low_nextday'])/df['close'] > 0.01,1,0)

    # 최근 거래량대비 거래량 증가여부 및 증감율
    # 전일, 최근3,7,10,20,30일 대비
    df['volume_last_1day'] = df['volume'].shift(1)
    df['volume_mean_3day'] = df['volume'].rolling(window=3).mean()
    df['volume_mean_7day'] = df['volume'].rolling(window=7).mean()
    df['volume_mean_10day'] = df['volume'].rolling(window=10).mean()
    df['volume_mean_20day'] = df['volume'].rolling(window=20).mean()
    df['volume_mean_30day'] = df['volume'].rolling(window=30).mean()

    df['vloume_inc_yn_vs_last_1day'] = np.where(df['volume'] > df['volume_last_1day'],1,0)
    df['vloume_inc_yn_vs_last_3day'] = np.where(df['volume'] > df['volume_mean_3day'],1,0)
    df['vloume_inc_yn_vs_last_7day'] = np.where(df['volume'] > df['volume_mean_7day'],1,0)
    df['vloume_inc_yn_vs_last_10day'] = np.where(df['volume'] > df['volume_mean_10day'],1,0)
    df['vloume_inc_yn_vs_last_20day'] = np.where(df['volume'] > df['volume_mean_20day'],1,0)
    df['vloume_inc_yn_vs_last_30day'] = np.where(df['volume'] > df['volume_mean_30day'],1,0)

    df['vloume_var_01_rate'] = ( df['volume'] - df['volume_last_1day'] ) / df['volume_last_1day']
    df['vloume_var_03_rate'] = ( df['volume'] - df['volume_mean_3day'] ) / df['volume_mean_3day']
    df['vloume_var_07_rate'] = ( df['volume'] - df['volume_mean_7day'] ) / df['volume_mean_7day']
    df['vloume_var_10_rate'] = ( df['volume'] - df['volume_mean_10day'] ) / df['volume_mean_10day']
    df['vloume_var_20_rate'] = ( df['volume'] - df['volume_mean_20day'] ) / df['volume_mean_20day']
    df['vloume_var_30_rate'] = ( df['volume'] - df['volume_mean_30day'] ) / df['volume_mean_30day']

    # 전일대비 증가여부
    df['close_last_1day'] = df['close'].shift(1)
    df['close_inc_yn_vs_last_1day'] = np.where(df['close'] > df['close_last_1day'],1,0)

    # 변동률 - 1시간(일)전, 4시간 전, 6시간 전, 12시간 전, 24시간 전
    df['close_var_01_rate'] = ( df['close'] - df['close'].shift(1) ) / df['close'].shift(1)
    df['close_var_04_rate'] = ( df['close'] - df['close'].shift(4) ) / df['close'].shift(4)
    df['close_var_06_rate'] = ( df['close'] - df['close'].shift(6) ) / df['close'].shift(6)
    df['close_var_12_rate'] = ( df['close'] - df['close'].shift(12) ) / df['close'].shift(12)
    df['close_var_24_rate'] = ( df['close'] - df['close'].shift(24) ) / df['close'].shift(24)
    
    # 변동폭 - 하이로우 차이
    df['highrow_rate_00'] = ( df['high'] - df['low'] ) / df['low']
    df['highrow_rate_01'] = ( df['highrow_rate_00'] - df['highrow_rate_00'].shift(1) ) / df['highrow_rate_00'].shift(1)
    df['highrow_rate_02'] = ( df['highrow_rate_00'] - df['highrow_rate_00'].shift(2) ) / df['highrow_rate_00'].shift(2)
    df['highrow_rate_03'] = ( df['highrow_rate_00'] - df['highrow_rate_00'].shift(3) ) / df['highrow_rate_00'].shift(3)
    
    df['highrow_rate_mean_03'] = df['highrow_rate_00'].rolling(window=3).mean()
    df['highrow_rate_mean_05'] = df['highrow_rate_00'].rolling(window=5).mean()
    df['highrow_rate_mean_10'] = df['highrow_rate_00'].rolling(window=10).mean()

    df['highrow_rate_04'] = ( df['highrow_rate_00'] - df['highrow_rate_mean_03'] ) / df['highrow_rate_mean_03']
    df['highrow_rate_05'] = ( df['highrow_rate_00'] - df['highrow_rate_mean_05'] ) / df['highrow_rate_mean_05']
    df['highrow_rate_06'] = ( df['highrow_rate_00'] - df['highrow_rate_mean_10'] ) / df['highrow_rate_mean_10']

    # 변동폭 - 로우종가 차이, 하이종가 차이
    df['closerow_rate_00'] = ( df['close'] - df['low'] ) / df['low']
    df['closerow_rate_01'] = df['closerow_rate_00'].shift(1)
    df['closerow_rate_02'] = df['closerow_rate_00'].shift(2)

    df['closerow_rate_mean_03'] = df['closerow_rate_00'].rolling(window=3).mean()
    df['closerow_rate_mean_05'] = df['closerow_rate_00'].rolling(window=5).mean()
    df['closerow_rate_mean_10'] = df['closerow_rate_00'].rolling(window=10).mean()
    
    df['highclose_rate_00'] = ( df['high'] - df['close'] ) / df['close']
    df['highclose_rate_01'] = df['highclose_rate_00'].shift(1)
    df['highclose_rate_02'] = df['highclose_rate_00'].shift(2)

    df['highclose_rate_mean_03'] = df['highclose_rate_00'].rolling(window=3).mean()
    df['highclose_rate_mean_05'] = df['highclose_rate_00'].rolling(window=5).mean()
    df['highclose_rate_mean_10'] = df['highclose_rate_00'].rolling(window=10).mean()

    # 최근 3일 종가평균 및 최근 3일평균종가 대비 증가여부 및 증가율
    df['close_mean_3day'] = ( df['close'].shift(1) + df['close'].shift(2) + df['close'].shift(3) ) / 3
    df['close_inc_yn_vs_last_3day'] = np.where(df['close'] > df['close_mean_3day'],1,0)
    df['close_var_rate_vs_last_3day'] = ( df['close'] - df['close_mean_3day'] ) / df['close_mean_3day']

    # 최근 10일 종가 생성
    for i in range(1,12):
        df['Var_{}'.format(i)] = df['close'].shift(i)

    # 최근 연속 상승일자(최근 10일 내)
    df['rec_up_his']=np.where(df['Var_1'] > df['Var_2'],
                    np.where(df['Var_2'] > df['Var_3'],
                    np.where(df['Var_3'] > df['Var_4'],
                    np.where(df['Var_4'] > df['Var_5'],
                    np.where(df['Var_5'] > df['Var_6'],
                    np.where(df['Var_6'] > df['Var_7'],
                    np.where(df['Var_7'] > df['Var_8'],
                    np.where(df['Var_8'] > df['Var_9'],
                    np.where(df['Var_9'] > df['Var_10'],
                    np.where(df['Var_10'] > df['Var_11'],10,9),8),7),6),5),4),3),2),1),0)

    # 최근 연속 하락일자(최근 10일 내)
    df['rec_down_his']=np.where(df['Var_1'] < df['Var_2'],
                    np.where(df['Var_2'] < df['Var_3'],
                    np.where(df['Var_3'] < df['Var_4'],
                    np.where(df['Var_4'] < df['Var_5'],
                    np.where(df['Var_5'] < df['Var_6'],
                    np.where(df['Var_6'] < df['Var_7'],
                    np.where(df['Var_7'] < df['Var_8'],
                    np.where(df['Var_8'] < df['Var_9'],
                    np.where(df['Var_9'] < df['Var_10'],
                    np.where(df['Var_10'] < df['Var_11'],10,9),8),7),6),5),4),3),2),1),0)

    # 이동평균선 활용 변수 생성
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA60'] = df['close'].rolling(window=60).mean()
    df['MA120'] = df['close'].rolling(window=120).mean()

    # 볼린저 밴드 활용한 종가 위치 변수
    df['bol_upper'] = df['MA20'] + ( 2 * df['close'].rolling(window=20).std() )
    df['bol_down'] = df['MA20'] - ( 2 * df['close'].rolling(window=20).std() )
    df['bol_upper_high'] = np.where( df['high'] - df['bol_upper']>0,1,0)
    df['bol_down_low'] = np.where( df['bol_down'] - df['low']>0,1,0)
    df['close_MA20_dif_std'] = ( df['MA20'] - df['close'] ) / df['close'].rolling(window=20).std()

    # 볼린저 밴드 이탈 횟수
    df['bol_upper_cnt_02'] = df['bol_upper_high'] + df['bol_upper_high'].shift(1)
    df['bol_upper_cnt_03'] = df['bol_upper_high'] + df['bol_upper_high'].shift(1) + df['bol_upper_high'].shift(2)
    df['bol_upper_cnt_04'] = df['bol_upper_high'] + df['bol_upper_high'].shift(1) + df['bol_upper_high'].shift(2) + df['bol_upper_high'].shift(3)
    df['bol_upper_cnt_05'] = df['bol_upper_high'] + df['bol_upper_high'].shift(1) + df['bol_upper_high'].shift(2) + df['bol_upper_high'].shift(3) + df['bol_upper_high'].shift(4)

    df['bol_down_cnt_02'] = df['bol_down_low'] + df['bol_down_low'].shift(1)
    df['bol_down_cnt_03'] = df['bol_down_low'] + df['bol_down_low'].shift(1) + df['bol_down_low'].shift(2)
    df['bol_down_cnt_04'] = df['bol_down_low'] + df['bol_down_low'].shift(1) + df['bol_down_low'].shift(2) + df['bol_down_low'].shift(3)
    df['bol_down_cnt_05'] = df['bol_down_low'] + df['bol_down_low'].shift(1) + df['bol_down_low'].shift(2) + df['bol_down_low'].shift(3) + df['bol_down_low'].shift(4)

    # 120선 기준 상승장여부
    df['trend_120_up_yn'] = np.where( df['MA120'] - df['MA120'].shift(1)>0,1,0)

    # 이동평균선 상승 배열 여부
    df['trend_up_yn'] = np.where( df['MA20'] > df['MA60'],
                        np.where( df['MA60'] > df['MA120'],1,0),0)

    # 지지 이동평균선 터치 후 상승여부
    # 10일, 20일, 60일, 120일
    df['touch_10_line'] = np.where( df['open'] > df['MA10'],
                        np.where( df['MA10'] > df['low'],
                        np.where( df['close'] > df['MA10'],1,0),0),0)
    df['touch_20_line'] = np.where( df['open'] > df['MA20'],
                        np.where( df['MA20'] > df['low'],
                        np.where( df['close'] > df['MA20'],1,0),0),0)
    df['touch_60_line'] = np.where( df['open'] > df['MA60'],
                        np.where( df['MA60'] > df['low'],
                        np.where( df['close'] > df['MA60'],1,0),0),0)
    df['touch_120_line'] = np.where( df['open'] > df['MA120'],
                        np.where( df['MA120'] > df['low'],
                        np.where( df['close'] > df['MA120'],1,0),0),0)

    # 저항 이동평균선 돌파여부
    # 10일, 20일, 60일, 120일
    df['break_10_line'] = np.where( df['open'] < df['MA10'],
                        np.where( df['close'] > df['MA10'],1,0),0)
    df['break_20_line'] = np.where( df['open'] < df['MA20'],
                        np.where( df['close'] > df['MA20'],1,0),0)
    df['break_60_line'] = np.where( df['open'] < df['MA60'],
                        np.where( df['close'] > df['MA60'],1,0),0)
    df['break_120_line'] = np.where( df['open'] < df['MA120'],
                        np.where( df['close'] > df['MA120'],1,0),0)

    # 이동평균선 기준 상하여부
    # 10일, 20일, 60일, 120일
    df['10_line_up'] = np.where( df['close'] > df['MA10'],1,0)
    df['20_line_up'] = np.where( df['close'] > df['MA20'],1,0)
    df['60_line_up'] = np.where( df['close'] > df['MA60'],1,0)
    df['120_line_up'] = np.where( df['close'] > df['MA120'],1,0)

    # Modeling
    df = df.drop(['open', 'high', 'low', 'close', 'value', 'close_nextday','high_nextday'
            , 'volume_last_1day', 'volume_mean_3day', 'volume_mean_7day', 'volume_mean_10day', 'volume_mean_20day', 'volume_mean_30day'
            , 'Var_1', 'Var_2', 'Var_3', 'Var_4', 'Var_5', 'Var_6', 'Var_7', 'Var_8', 'Var_9', 'Var_10', 'Var_11'
            , 'MA10','MA20','MA60','MA120'
            , 'bol_upper', 'bol_down'],axis=1)

    # 결측치 제거
    df = df.dropna()
    feature_columns = list(df.columns.difference(['up_yn','down_yn']))
    X = df[feature_columns]
    y = df['up_yn']
    y1 = df['down_yn']
    
    # 정규화
    X = MinMaxScaler(X)
    
    # 상승 예측모델링 수행
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=12)

    lgbm_wrapper=LGBMClassifier(n_estimators=300,
    #        num_leaves= 71,
            max_depth= 10,
    #        scale_pos_weight= 970,
    #        min_child_weight= 55,
            subsample= 0.6,
            colsample_bytree= 0.6)
    evals=[(X_test,y_test)] #검증단계
    lgbm_wrapper.fit(X_train,y_train,early_stopping_rounds=100,eval_metric='logloss',eval_set=evals,verbose=True)

    preds=lgbm_wrapper.predict(X_test)
    pred_proba=lgbm_wrapper.predict_proba(X_test)[:,1]
    
    preds_X=lgbm_wrapper.predict(X)
    up_yn = preds_X[-1]

    # 하락 예측모델링 수행
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X,y1,test_size=0.3, random_state=12)

    lgbm_wrapper1=LGBMClassifier(n_estimators=200,
    #        num_leaves= 71,
            max_depth= 10,
    #        scale_pos_weight= 970,
    #        min_child_weight= 55,
            subsample= 0.6,
            colsample_bytree= 0.6)
    evals1=[(X_test1,y_test1)] #검증단계
    lgbm_wrapper1.fit(X_train1,y_train1,early_stopping_rounds=100,eval_metric='logloss',eval_set=evals1,verbose=True)

    preds1=lgbm_wrapper1.predict(X_test1)
    pred_proba1=lgbm_wrapper1.predict_proba(X_test1)[:,1]
    
    preds_X1=lgbm_wrapper1.predict(X)
    down_yn = preds_X1[-1]

    return up_yn, X_train, y_test, preds, pred_proba, down_yn, y_test1, preds1, pred_proba1

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
# 수정된 get_clf_eval() 함수 
def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix( y_test, pred)
    accuracy = accuracy_score(y_test , pred)
    precision = precision_score(y_test , pred)
    recall = recall_score(y_test , pred)
    f1 = f1_score(y_test,pred)
    # ROC-AUC 추가 
    roc_auc = roc_auc_score(y_test, pred_proba)
    print('오차 행렬')
    print(confusion)
    # ROC-AUC print 추가
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f},\
    F1: {3:.4f}, AUC:{4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))

def post_message(token, channel, text):
    """슬랙 메시지 전송"""
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel,"text": text}
    )

def get_target_price(ticker):
    """변동성 돌파 전략으로 매수 목표가 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="minutes30", count=2)
    target_price = df.iloc[0]['close'] * 0.998
    return target_price

def get_low_price(ticker):
    """변동성 돌파 전략으로 매수 목표가 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="minutes60", count=1)
    low_price = df.iloc[0]['low']
    return low_price

def get_start_time(ticker):
    """시작 시간 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="minutes30", count=1)
    start_time = df.index[0]
    return start_time

def get_ma15(ticker):
    """15일 이동 평균선 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="day", count=15)
    ma15 = df['close'].rolling(15).mean().iloc[-1]
    return ma15

def get_balance(ticker):
    """잔고 조회"""
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == ticker:
            if b['balance'] is not None:
                return float(b['balance'])
            else:
                return 0
    return 0

def get_current_price(ticker):
    """현재가 조회"""
    return pyupbit.get_orderbook(tickers=ticker)[0]["orderbook_units"][0]["ask_price"]

# 로그인
upbit = pyupbit.Upbit(access, secret)
print("autotrade start")

# 시작 메세지 슬랙 전송
post_message(myToken,"#stock", "autotrade start")


now = datetime.datetime.now()
start_time = get_start_time("KRW-BTC")
end_time_model = start_time + datetime.timedelta(seconds=10)
end_time = start_time + datetime.timedelta(minutes=20)
end_time2 = start_time + datetime.timedelta(minutes=29) + datetime.timedelta(seconds=55)
BTC = get_balance("BTC")

up_yn, X_train, y_test, preds, pred_proba, down_yn, y_test1, preds1, pred_proba1 = Light_GBM("KRW-BTC")

target_price = get_target_price("KRW-BTC")
current_price = get_current_price("KRW-BTC")
low_price = get_low_price("KRW-BTC")
krw = get_balance("KRW")
order_cnt = round(krw / target_price, 7) 
avg_buy_price = upbit.get_avg_buy_price("BTC")
buy_price = math.floor(math.floor(target_price) / 1000) * 1000
sell_price = math.ceil(math.ceil(avg_buy_price * 1.005) / 1000) * 1000

get_clf_eval(y_test, preds, pred_proba)
get_clf_eval(y_test1, preds1, pred_proba1)
print(now)
print(up_yn)
print(down_yn)
print(current_price)
