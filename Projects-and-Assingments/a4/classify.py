
# coding: utf-8

# In[47]:

from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def accuracy(truth, predicted):
    return len(np.where(truth==predicted)[0]) / len(truth)

def main():
    tweetr=pickle.load(open('USTweets.pkl', 'rb'))
    TW=[]
    for i in range(len(tweetr)):
        if('RT' not in tweetr[i]['text']):
            TW.append(tweetr[i]['text'])
    train_tweets = pd.read_csv('SS.csv',names=['ItemID', 'Sentiment', 'SentimentSource','SentimentText'])
    print(Counter(list(train_tweets["Sentiment"])))
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,1),use_idf=True)
    X = vectorizer.fit_transform(train_tweets['SentimentText'])
    print('vectorized %d training tweets. found %d vocab terms.' % (X.shape[0], X.shape[1]))
    y = np.array(train_tweets['Sentiment'])
    vocab = np.array(vectorizer.get_feature_names())
    X1 = vectorizer.transform(TW)
    print('transformed %d testing tweets with %d terms.' % (X1.shape[0], X1.shape[1]))
    
    model = LogisticRegression()
    model.fit(X, y)
    
    predicted = model.predict(X)
    print('accuracy on training data=%.3f' % accuracy(y, predicted))
    
    a=model.predict_proba(X1)
    b=model.predict(X1)
    indexes=['Negative','Positive']
    positive=[]
    negative=[]
    pred=[]
    for c,i in enumerate(a):
        i=list(i)
        if( b[c]==0):
            positive.append((TW[c],indexes[b[c]],max(i),list(zip(indexes,i))))
    for c,i in enumerate(a):
        i=list(i)
        if( b[c]==1):
            negative.append((TW[c],indexes[b[c]],max(i),list(zip(indexes,i))))
    pred.append(negative)
    pred.append(positive)
    pickle.dump(pred, open('Classify.pkl', 'wb'))
    coef = model.coef_[0]
    # Sort them in descending order.
    top_coef_ind = np.argsort(coef)[::-1][:10]
    # Get the names of those features.
    top_coef_terms = vocab[top_coef_ind]
    # Get the weights of those features
    top_coef = coef[top_coef_ind]
    # Print the top 10.
    print('top weighted terms for positive class:')
    print([x for x in zip(top_coef_terms, top_coef)])
    top_coef_ind = np.argsort(coef)[::1][:10]
    top_coef_terms = vocab[top_coef_ind]
    top_coef = coef[top_coef_ind]
    # Print the top 10.
    print('top weighted terms for negative class:')
    print([x for x in zip(top_coef_terms, top_coef)])
    print("done")
    
    
if __name__ == '__main__':
    main()

