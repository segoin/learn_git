import time
import pyupbit
import datetime
import requests
import json
import numpy as np
import pandas as pd
import pyupbit
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import math 

access = "ccyUip4ujykayb5qFP9xkncrbbt9HPM98vtVQeqV"
secret = "x3fUq5n4MMHDyKDKeNZdVOcoMjiWQqB0pBBErDjn"
myToken = "xoxb-1702501177444-1689568357894-eTntzP8uGJlB0sDjbpfDP5mK"

def post_message(token, channel, text):
    """슬랙 메시지 전송"""
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel,"text": text}
    )
    
def post_message1(channel, text): 
    SLACK_BOT_TOKEN = "xoxb-1702501177444-1689568357894-eTntzP8uGJlB0sDjbpfDP5mK"
    headers = {
        'Content-Type': 'application/json', 
        'Authorization': 'Bearer ' + SLACK_BOT_TOKEN
        }
    payload = {
        'channel': channel,
        'text': text
        }
    r = requests.post('https://slack.com/api/chat.postMessage', 
        headers=headers, 
        data=json.dumps(payload)
        )

print("autotrade start")
post_message(myToken,"#stock", "autotrade start")
post_message1("#stock", "Hello World!")

