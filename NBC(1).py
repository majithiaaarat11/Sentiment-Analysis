# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 21:24:30 2018

@author: majit
"""

import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
import numpy as np

df = pd.read_csv("twitter.csv",encoding = "ISO-8859-1")

stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf = True, lowercase = True, strip_accents = 'ascii', stop_words = stopset)

y=df.Sentiment

X = vectorizer.fit_transform(df.SentimentText)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=50)

clf=naive_bayes.MultinomialNB()
clf.fit(X_train,y_train)

roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])

review = input("Enter a review: ")

review_array = np.array([review])
review_vector = vectorizer.transform(review_array)
print(clf.predict(review_vector))