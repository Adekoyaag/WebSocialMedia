# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:13:57 2022

@author: HP
"""
from matplotlib import pyplot as plt
import pandas as pd
import random
df = pd.read_csv("worldcup.csv")
text =  df['Text'].dropna()

#wordcloud
import nltk
from nltk.corpus import stopwords
import re
import networkx
from textblob import TextBlob

text = pd.DataFrame(text)

# Create textblob objects of the tweets
sentiment_objects = [TextBlob(tweet) for tweet in text]

sentiment_objects[0].polarity, sentiment_objects[0]

text['polarity'] = text['Text'].apply(lambda text: TextBlob(text).sentiment.polarity)
text['subjectivity'] = text['Text'].apply(lambda text: TextBlob(text).sentiment.subjectivity)

def sentiment_col(y):
    if y < 0 :
        return 'negative'
    elif y == 0:
        return 'neutral'
    else :
        return 'positive'
text['polarity'] = text['polarity'].apply(sentiment_col)

#define data
df2 = text.groupby('polarity').size().reset_index(name='coun')
n = df2['polarity'].unique().__len__()+1
all_colors = list(plt.cm.colors.cnames.keys())
random.seed(100)
c = random.choices(all_colors, k=n)

# Plot Bars
plt.figure(figsize=(10,5), dpi= 80)
plt.bar(df2['polarity'], df2['coun'], color=c, width=.5)
for i, val in enumerate(df2['coun'].values):
    plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':25})

# Decoration
plt.gca().set_xticklabels(df2['polarity'], rotation=60, horizontalalignment= 'right')
plt.title("Ronaldo's Sentiment Analysis", fontsize=10)
plt.ylabel('Number of Devices', fontsize=10)
plt.ylim(0, 2000)
plt.show()
 


 
#wordcloud

import re
import nltk

a = text.to_string() #loads the row from dataframe
print(a)
regex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
match = re.sub(regex_pattern,'',a) #replaces pattern with ''
print(match)
#removing url
pattern = re.compile(r'(https?://)?(www\.)?(\w+\.)?(\w+)(\.\w+)(/.+)?')
match = re.sub(pattern,'',a)
print(match)
#


match = re.sub("@[A-Za-z0-9_]+","", match)
match = re.sub("#[A-Za-z0-9_]+","", match)


print(type(match))
corpus = match
print(a)
sentences = nltk.sent_tokenize(corpus)

print(type(sentences))

sentence_tokens = ""
for sentence in sentences:
    sentence_tokens += sentence
    


#word tokenization
words=nltk.word_tokenize(sentence_tokens)
print(words)
for word in words:
    print(word)
    
import nltk
from nltk.corpus import stopwords
print(stopwords.words('english'))

#stop word removal
stop_words = set(stopwords.words('english'))
filtered_words=[]

for w in words:
    if w not in stop_words:
        filtered_words.append(w)

print('/n With stop words:', words)
print('/n After removing stop words:', filtered_words)


#finding the frequency distribution of words
frequency_dist = nltk.FreqDist(filtered_words)

#SORTING THE FREQUENCY DISTRIBUTION
sorted(frequency_dist,key=frequency_dist.__getitem__,reverse=True)[0:30]

#Keeping only the large words(more than 3 characters)
large_words = dict([(k,v) for k,v in frequency_dist.items() if len (k)>3])

frequency_dist = nltk.FreqDist(large_words)
frequency_dist.plot(30, cumulative=False)

#Visualising the distribution of words using matplotlib and wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt


wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="black").generate_from_frequencies(frequency_dist)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


