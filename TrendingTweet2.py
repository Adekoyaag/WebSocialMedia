# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 09:51:16 2022

@author: DELL
"""

import tweepy
import matplotlib.pyplot as plt



from twython import Twython
import json


import json
    #load crendentials from json

with open("twitter_credentials.json","r") as file:
    creds=json.load(file)
#from twitter import Twitter




consumer_key = ''
consumer_secret = ''
access_token = ''
access_secret = ''



auth = tweepy.OAuthHandler(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
auth.set_access_token(creds['ACCESS_TOKEN'], creds['ACCESS_SECRET'])
our_api = tweepy.API(auth)

#(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'], creds['ACCESS_TOKEN'], creds['ACCESS_SECRET'])#


#python_tweets = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])



trends= our_api.get_place_trends(id=1398823)
print(trends)


import pandas as pd



dict_={'Name':[], 'Tweet_volume':[]}



for values in trends:
    
    for trend in values['trends']:
        
        dict_['Name'].append(trend['name'])
        
        dict_['Tweet_volume'].append(trend['tweet_volume'])


df=pd.DataFrame(dict_)
#df.sort_values(by='favorite_count', inplace=True, ascending=False)
df = df[df["Tweet_volume"]>0] #Filter to remove all NaN tweet_volumes
df.head(5)



df.plot.bar(x="Name", y="Tweet_volume", rot=70, title="Top Trends and Tweet Volume in the Ukraine");



plt.show(block=True);