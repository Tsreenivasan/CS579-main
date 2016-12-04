
# coding: utf-8


from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import configparser
import requests
import pickle
from TwitterAPI import TwitterAPI,TwitterRestPager


def get_twitter():
    config = configparser.ConfigParser()
    config.read('Twitter_tokens.cfg')
    consumer_key = config.get('twitter','consumer_key')
    consumer_secret = config.get('twitter','consumer_secret')
    access_token = config.get('twitter','access_token')
    access_token_secret = config.get('twitter','access_token_secret')
    twitter = TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
    return twitter


def get_census_names():
    """ Fetch a list of common male/female names from the census.
    For ambiguous names, we select the more frequent gender."""
    males = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.male.first').text.split('\n')
    females = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.female.first').text.split('\n')
    males_pct = dict([(m.split()[0].lower(), float(m.split()[1]))
                  for m in males if m])
    females_pct = dict([(f.split()[0].lower(), float(f.split()[1]))
                    for f in females if f])
    male_names = set([m for m in males_pct if m not in females_pct or
                  males_pct[m] > females_pct[m]])
    female_names = set([f for f in females_pct if f not in males_pct or
                  females_pct[f] > males_pct[f]])    
    return male_names, female_names

def get_first_name(tweet):
    if 'user' in tweet and 'name' in tweet['user']:
        parts = tweet['user']['name'].split()
        if len(parts) > 0:
            return parts[0].lower()

def sample_tweets(twitter, limit, male_names, female_names,tweets,trend_term):
    while True:
        try:
            # Restrict to U.S.
            for response in twitter.request('statuses/filter',
                        {'track':trend_term}):
                if 'user' in response:
                    name = get_first_name(response)
                    if ((name in male_names or name in female_names) and ('RT' not in response["text"])):
                        tweets.append(response)
                        if len(tweets) % 10 == 0:
                            print('found %d tweets' % len(tweets))
                            pickle.dump(tweets, open('USTweets.pkl', 'wb'))
                            print("Press cancel any time to proceed with the next steps with the datat collected so far. ")
                if len(tweets) >= limit:
                    return tweets
        except KeyboardInterrupt:
            print(len(tweets)," Streams Collected")
            break
    return tweets


def main():
    api= get_twitter()
    print('Established Twitter connection.')
    male_names, female_names = get_census_names()
    print('found %d female and %d male names' % (len(male_names), len(female_names)))
    print('male name sample:', list(male_names)[:5])
    print('female name sample:', list(female_names)[:5])
    tweets=[]
    tweet_limit=5000
    print("streaming tweets on a trending tag ",list(api.request('trends/place',{'id': '2379574'}))[0]["name"])
    tweets = sample_tweets(api, tweet_limit, male_names, female_names,tweets,list(api.request('trends/place',{'id': '2379574'}))[0]["name"])
    if(len(tweets)>0):
        pickle.dump(tweets, open('USTweets.pkl', 'wb'))
    print('sampled %d tweets' % len(tweets))
    print('top names:', Counter(get_first_name(t) for t in tweets).most_common(10))
    print("Kindly execute the cluster.py and the classify.py files for network analysis and tweet sentiment analysis for the sample records")
        
if __name__ == '__main__':
    main()




