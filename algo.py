import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from csv import writer
from csv import reader
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from stop_words import get_stop_words
import joblib
warnings.simplefilter('ignore')

labels = pd.read_csv('labels.csv')

for i, row in labels.iterrows():

    tweet = str(row["tweet"])

    if tweet.find(';') != -1 :
        tweet = tweet.replace (";"," ")
    
    if tweet.find('"') != -1:
        tweet = tweet.replace ('"'," ")

    if tweet.find('&') != -1 :
        tweet = ' '.join(filter(lambda x:x[0]!='&', tweet.split()))

    while tweet.find('&') != -1 :
        a = tweet.find('&') -1
        tweet = tweet[:(a+1)] + " " + tweet[(a+1):]
        tweet = ' '.join(filter(lambda x:x[0]!='&', tweet.split()))
    
    labels.at[i, "tweet"] = tweet

from sklearn.preprocessing import LabelEncoder
X = labels['tweet']
y = labels[['class']]

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = make_pipeline(
    TfidfVectorizer(stop_words=get_stop_words('en')),
    OneVsRestClassifier(SVC(kernel='linear', probability=True))
)
clf.fit(X_train,y_train)

joblib.dump(clf,"model.pkl")

