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
from sklearn.metrics import confusion_matrix
import numpy as np

df = pd.read_csv("https://storage.googleapis.com/kaggle-competitions-data/kaggle/2558/training.txt?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1535135533&Signature=MuaM3iscrkhpruPU6OlyVPn1LOG9Gbj8C4%2FfPcsR76XDl3O9rB5DnpONGXs3gl0aUNbfkWDXSORryNPUKPVa5ky2gow5cdNVuSf%2B%2BfgOv1LKn1eJRiLYzmMD%2FdAYn6ksj4Dw0tlS6XozhtsMjjSO6%2FthcxyIoYHqyXIpkv6B7iZ6bBmoVqXOTlxL27y3ej965R9QBHPVIZTKLoviOmCtW8560v%2FS%2FJxaJFLgZGhAf%2Fg%2BF%2FfX1B9%2Bn0VMSdlG5oa%2Fh6VE%2FWdp0Mq3%2F2oqE9nz0fSO5Jwp4VjZGZCP8HWFGnLN%2BLG3ERo9AiEKDDcKUm%2BN00Vf4ua1a%2Fc9qm18sq68cA%3D%3D",sep="\t",names=['liked','review'])

stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf = True, lowercase = True, strip_accents = 'ascii', stop_words = stopset)

y=df.liked

X = vectorizer.fit_transform(df.review)

print(X.shape)
print(y.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

clf=naive_bayes.MultinomialNB()
clf.fit(X_train,y_train)

roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])
confusion_matrix(y_test,clf.predict(X_test)[:,1])

review = input("Enter a review: ")

review_array = np.array([review])
review_vector = vectorizer.transform(review_array)
print(clf.predict(review_vector))
