# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:24:03 2022

@author: DELL
"""

#news api
import pprint
import requests
secret = '544f878df5ae43cbb3b1716322f09c9a'

url = 'https://newsapi.org/v2/everything?'
parameters = {
    'q': 'cristiano ronaldo', # query phrase
    'pageSize': 50,  # maximum is 100
    'apiKey': secret # your own API key
}
response = requests.get(url, params=parameters)

# Convert the response to JSON format and pretty print it
response_json = response.json()
pprint.pprint(response_json)

output  = response_json['articles']
List_Url = []
for i in range (1 , 6):
    List_Url.append(output[i]['url'])
    i+=1
    
for i in response_json['articles']:
    print(i['title'])    
    
from wordcloud import WordCloud
import nltk
from textblob import TextBlob
import tkinter as tk
from newspaper import Article

for i in range(0 , len(List_Url)):

    article = Article(List_Url[i])
    article.download()
    article.parse()
    article.nlp()
    print(f'Title:{article.title}')
    print(f'Authors:{article.authors}')
    print(f'Publication Date:{article.publish_date}')
    print(f'Summary:{article.summary}')
    analyse = TextBlob(article.text)
    print(analyse.sentiment)
    i+= 1    
    
from wordcloud import WordCloud
import nltk
from textblob import TextBlob
import tkinter as tk
from newspaper import Article
Text_List=[]
for i in range(0 , len(List_Url)):

    article = Article(List_Url[i])
    article.download()
    article.parse()
    article.nlp()
    Text_List.append(article.text)  
    
import nltk
for i in range(0 , 5):
    Text_List[i]
    sentences = nltk.sent_tokenize(Text_List[i])
    Words= nltk.word_tokenize(Text_List[i])
    j=0
    for sentence in sentences:
        j+=1
    print("Article sentences:",i , j )
    K=0
    for word in Words:
        K+=1
    print("Article Words", i , K)
    i+=0

from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import wordcloud
import pandas as pd
stop_wrods = set(stopwords.words("english"))

Words =nltk.word_tokenize(Text_List[3])
filtered_words= []
for w in Words:
    if w not in stop_wrods:
        filtered_words.append(w)

for i in range(0 , len(filtered_words)):
    filtered_words[i] = filtered_words[i].lower()

frequency_dist = nltk.FreqDist(filtered_words)
sorted(frequency_dist, key=frequency_dist.__getitem__, reverse= True)[0:30]
large_words = dict([(k,v) for k , v in frequency_dist.items() if len(k) > 3 ] )

frequency_dist = nltk.FreqDist(large_words)
frequency_dist.plot(30,cumulative=False)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud(max_font_size=50, max_words=100,
background_color="black").generate_from_frequencies(frequency_dist)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
documents = Text_List[3].splitlines()
count_vectorizer = CountVectorizer()
bag_of_words =count_vectorizer.fit_transform(documents)
feature_names = count_vectorizer.get_feature_names()
print(pd.DataFrame(bag_of_words.toarray(), columns=feature_names))



from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
tfidf_vectorizer = TfidfVectorizer()
values = tfidf_vectorizer.fit_transform(documents)
feature_names =tfidf_vectorizer.get_feature_names()
TFIDF_matrix = pd.DataFrame(values.toarray() , columns= feature_names)
print(TFIDF_matrix)

#save to csv
import pandas as pd
import re
LDA =[]
for i in range (0 , len(Text_List)):

    x = re.sub(r'[^\w]', ' ', Text_List[i])
    x = re.sub(r'[\d]', ' ', x )
    x = re.sub(' +', ' ', x)
    LDA.append(x)
    i+=1
df = pd.DataFrame(LDA) 
    
# saving the dataframe 
df.to_csv('Articles.csv')
###########ist############
import re
import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
def load_data(path, file_name):
    documents_list=[]
    titles=[]
    with open(os.path.join(path, file_name) , "r" , encoding = 'utf-8') as fin:
        for line in fin.readlines():
            text = line.strip()
            documents_list.append(text)
    print("Total Number of Dcouments: " , len(documents_list))
    titles.append(text[0:min(len(text) , 100)])
    return documents_list, titles

def preprocess_data(doc_set):
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    Texts=[]
    for i in doc_set:
        raw=i.lower()
        tokens= tokenizer.tokenize(raw)
        stopped_tokens= [i for i in tokens if not i in en_stop]
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        Texts.append(stemmed_tokens)
    
    return Texts

def prepare_corpus(doc_clean):
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix= [ dictionary.doc2bow(doc) for doc in doc_clean]
    return dictionary, doc_term_matrix


def create_gensim_lda_model(doc_clean , number_of_topics, words):
    dictionary, doc_term_matrix = prepare_corpus(doc_clean)
    lsamodel = LsiModel(doc_term_matrix , num_topics=number_of_topics , id2word= dictionary)
    print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return lsamodel

    
def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def plot_graph(doc_clean,start, stop, step):
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix,doc_clean,
                                                            stop, start, step)
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
#plot_graph(clean_text ,start,stop,step)

start,stop,step=2,12,1
##plot_graph(clean_text ,start,stop,step)
number_of_topics=5
words= 10
document_list, titles= load_data("C:/Users/DELL/Documents/web_analaytics", "Articles.csv")
clean_text = preprocess_data(document_list)
model = create_gensim_lda_model(clean_text, number_of_topics, words)
plot_graph(clean_text ,start,stop,step)
##########################