
# coding: utf-8

# In[63]:

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

def get_twitter():
    config = configparser.ConfigParser()
    config.read('Twitter_tokens.cfg')
    consumer_key = config.get('twitter','consumer_key')
    consumer_secret = config.get('twitter','consumer_secret')
    access_token = config.get('twitter','access_token')
    access_token_secret = config.get('twitter','access_token_secret')
    twitter = TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
    #twitter = TwitterAPI('Z9nbyf4jhEDqOIo0eSOBGVXcy','xY9nCLesJqICteGIKdt1uYsXtOQsyMqYgcSJDJJ6rKg2A7Rkwg','365484690-dMMg9kiLIhqlRI843GAdbzi9g0keBMzpDCw8JyE4','E4V9y24mYfRLLJ2Q2lOY4FAnthCqzva7IQVIlk0IhcODp')
    return twitter

def Get_Friends(api,D,tweets):
    j=len(D)
    for i in range(len(tweets)):
        alist=[]
        aa1={}
        try:
            r = api.request('friends/ids',{'screen_name': tweets[i]["user"]["screen_name"],'count':5000})
            for item in r:
                alist.append(item)
            aa1['friends']=alist
            aa1['screen_name']=tweets[i]["user"]["screen_name"]
            aa1['tweet']=tweets[i]["text"]
            D[j]=dict(aa1)
            j+=1
        except:
            if(len(D)>0):
                print("Data Collected for ",len(D)," Users")
            else:
                print("Twitter Timeout;Data Collection failed")
                sys.exit(0)
            break;
    return D

def get_common_Friends(dct):
    groups = defaultdict(list)
    finallist=[]
    for r in range(len(dct)):
        for rs in range(len(dct)):
            prelist=[]
            count=0
            if dct[rs]['screen_name']==dct[r]['screen_name']:
                #break
                pass 
            else:
                a=set(dct[r]['friends'])&set(dct[rs]['friends'])
                if len(a)!=0:
                    prelist.append(dct[r]['screen_name'])
                    prelist.append(dct[rs]['screen_name'])
                    prelist.sort()
                    prelist.append(a)
                    finallist.append(prelist)
    finallist.sort()
    unique = []
    [unique.append(item) for item in finallist if item not in unique]
    return unique

def create_graph(users):
    G=nx.Graph()

    for in1 in users:
        prev1=''
        prev2=''
        prev3=[]
        count=0
        for in2 in in1:
            if (type(in2)is str):
                G.add_node(in2)
                if len(prev1)!=0:
                    if(prev2!=prev1):
                        G.add_edge(in2,prev1)
                        prev2=in2
                if(len(prev1)==0):
                    prev1=in2
            elif (type(in2)is set):
                prev3=in2
                for in3 in in2:
                    G.add_node(in3)
        if(len(prev1)!=0 and len(prev2)!=0 and len(prev3)!=0):
            for i in prev3:
                G.add_edge(i,prev1)
                G.add_edge(i,prev2)

    print('graph with N=%d E=%d' % (len(G.nodes()), len(G.edges())))#nx.draw(G,with_labels=True)
    pos=nx.spring_layout(G)
    #nx.draw_networkx(G,pos,node_size=60,width=0.2,alpha=.5,with_labels=False)
    #plt.show() # display
    return G

def partition_girvan_newman(graph,partition_value):

    graph.remove_edges_from(graph.selfloop_edges())
    components = [c for c in nx.connected_component_subgraphs(graph)]
    last=len(components)
    print("Creating Cluster::",last)
    while(len(components)<partition_value):
        betweenness=nx.edge_betweenness_centrality(graph)
        a=max(betweenness, key=betweenness.get)
        graph.remove_edge(*a)
        components = [c for c in nx.connected_component_subgraphs(graph)]
        if(last!=len(components)):
            last=len(components)
            print("Creating Cluster::",last)
    return graph



def main():
    api= get_twitter()
    print('Established Twitter connection.')
    tweets=pickle.load(open('USTweets.pkl', 'rb'))
    print(len(tweets)," Tweets read.")
    D=defaultdict()
    users_list=[]
    for i in tweets:
        users_list.append(i["user"]["screen_name"])
    D=Get_Friends(api,D,tweets)
    pickle.dump(D, open('Friend_dict.pkl', 'wb'))
    ret_list=get_common_Friends(D)
    G=create_graph(ret_list)
    a=partition_girvan_newman(G.copy(),5)
    print("Partition done")
    pos=nx.spring_layout(a.copy(),iterations=300)
    components = [c for c in nx.connected_component_subgraphs(a.copy())]
    colors=["b","g","r","orange","brown","purple","pink"]
    length=len(components)
    for i in range(len(components)):
        nx.draw_networkx(components[i],pos,node_color=colors[i],node_size=40,width=0.2,alpha=.7,with_labels=False)
    plt.clf() 
    pos=nx.spring_layout(a.copy(),iterations=300)
    for i in range(len(components)):
        nx.draw_networkx(components[i],pos,node_color=colors[i],node_size=40,width=0.2,alpha=.7,with_labels=False)
    node_count=0
    for i in range(length):
        node_count+=len(components[i].nodes())
    print("Total users collected::",len(D))
    print("Average number of users per community::",node_count/length)
    pickle.dump(a, open('Cluster.pkl', 'wb'))
    plt.savefig("oppp.png",dpi=1200)
    plt.show()
    print("Clusters for Pre-collected data")
    time.sleep(5)
    a=pickle.load(open('Pre_Cluster.pkl', 'rb'))
    components = [c for c in nx.connected_component_subgraphs(a.copy())]
    pos=nx.spring_layout(a.copy(),iterations=100)
    for i in range(len(components)):
        nx.draw_networkx(components[i],pos,node_color=colors[i],node_size=40,width=0.2,alpha=.7,with_labels=False)
    length=len(components)
    node_count=0
    for i in range(length):
        node_count+=len(components[i].nodes())
    print("Total users collected::",30)
    print("Average number of users per community::",node_count/length)
    plt.savefig("oppp1.png",dpi=1200)
    plt.show() 
        
        
if __name__ == '__main__':
    main()


