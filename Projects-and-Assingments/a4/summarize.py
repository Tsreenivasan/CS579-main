
# coding: utf-8

# In[19]:

from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import configparser
import pickle
from TwitterAPI import TwitterAPI,TwitterRestPager
from itertools import groupby
from collections import defaultdict


tweets=pickle.load(open('USTweets.pkl', 'rb'))
users_list=[]
for i in tweets:
        users_list.append(i["user"]["screen_name"])
print("Number of messages collected:",len(tweets))

Dict=pickle.load(open('Friend_dict.pkl', 'rb'))
print("Number of users collected:",len(Dict))

a=pickle.load(open('Cluster.pkl', 'rb'))
components = [c for c in nx.connected_component_subgraphs(a.copy())]
length=len(components)
node_count=0
for i in range(length):
    node_count+=len(components[i].nodes())
print("Average number of users per community::",node_count/length)
C=pickle.load(open('Classify.pkl', 'rb'))
print("Tweets classified as Positive::",len(C[0]))
print("One tweet from the Positive class is below::\n",sorted(C[0],key=lambda x:x[2],reverse=True)[0])
print("Tweets classified as Negative::",len(C[1]))
print("One tweet from the Negative class is below::\n",sorted(C[1],key=lambda x:x[2],reverse=True)[0])

