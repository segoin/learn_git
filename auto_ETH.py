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

post_message(myToken,"#stock", "autotrade start")

