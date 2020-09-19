import requests
from bs4 import BeautifulSoup
from newspaper import Article
import newspaper
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import nltk.classify.util
import csv
from nltk.corpus import movie_reviews
import pandas as pd
import nltk
import random
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import classifiers


##########################SCRAPE ARTICLES##############################################################

ticker='Trump'
articles_examined=50
prefix='https://news.google.com/'
url='https://news.google.com/search?q='+ticker+'&hl=en-US&gl=US&ceid=US%3Aen'
r1 = requests.get(url)
coverpage = r1.content
soup1 = BeautifulSoup(coverpage, 'html5lib')
coverpage_news = soup1.find_all('div', class_="NiLAwe y6IFtc R7GTQ keNKEd j7vNaf nID9nc")
links=[]
for article in (coverpage_news):
    links.append(prefix+article.a["href"])

titles=[]
texts=[]
summaries=[]
counter=0
for link in links:
    try:
        url=link
        article = Article(url, language="en")
        article.download() 
        article.parse() 
        article.nlp() 
        titles.append(article.title) #prints the title of the article 
        texts.append((article.text)) #prints the entire text of the article
        summaries.append(article.summary) #prints the summary of the article
        #print(article.keywords) #prints the keywords of the article
        counter+=1
        if counter>=articles_examined:
            break
            
    except newspaper.article.ArticleException:
        continue

######################PERSONAL TRAINER, YOU CAN ADD WHATEVER TEXT YOU WANT TO TRAIN THE CLASSIFIER ON########
'''
def word_feats(words):
    return dict([(word, True) for word in words])
negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')
negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
negcutoff = int(len(negfeats)*3/4)
poscutoff = int(len(posfeats)*3/4)
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]


classifier = classifiers.NaiveBayesClassifier(trainfeats)
'''
cl='Naive Bayes'

import pickle
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

text_counter=0
texts_neg_sum=[]
texts_pos_sum=[]
result_te=''
for text in texts:
    prob_dist = classifier.prob_classify(text)
    texts_pos_sum.append(round(prob_dist.prob("pos"), 2))
    texts_neg_sum.append(round(prob_dist.prob("neg"), 2))
    text_counter+=1

if sum(texts_neg_sum)>sum(texts_pos_sum):
    result_te='negative'
elif sum(texts_neg_sum)<sum(texts_pos_sum):
    result_te='positive'
   
n_sent=((sum(texts_neg_sum)/text_counter)*100)
p_sent=((sum(texts_pos_sum)/text_counter)*100)


##FINDING SUMMARY OF ARTICLE WITH SENTIMENT NEAREST TO AVERAGE###

sent_list=[]
avg_num=0
if sum(texts_neg_sum)>sum(texts_pos_sum):
    sent_list=texts_neg_sum
    avg_num=n_sent
elif sum(texts_neg_sum)<sum(texts_pos_sum):
    sent_list=texts_pos_sum
    avg_num=p_sent

clossest_sent=min(sent_list, key=lambda x:abs(x-avg_num))
avg_summary=summaries[sent_list.index(clossest_sent)]



######################PRE-TRIANED CLASSIFIER, TRAINED ON THE VADER LEXICON###################################

import nltk
#nltk.download('vader_lexicon') # do this once: grab the trained model from the web
from nltk.sentiment.vader import SentimentIntensityAnalyzer

Analyzer = SentimentIntensityAnalyzer()
cla='SentimentIntensityAnalyzer'

neg_sum=0
pos_sum=0
neu_sum=0
counter=0
result=''
for text in texts:
    a=Analyzer.polarity_scores(text)
    neg_sum+=(a['neg'])
    neu_sum+=(a['neu'])
    pos_sum+=(a['pos'])
    counter+=1

if neg_sum>pos_sum:
    result='negative'
elif neg_sum<pos_sum:
    result='positive'

neg_sent=((neg_sum/counter)*100)
pos_sent=((pos_sum/counter)*100)
neu_sent=((neu_sum/counter)*100)


##########MARKET BEAT SCRAPED SENTIMENT ANALYZER#################
'''
market_beat=''
url='https://www.marketbeat.com/stocks/NASDAQ/'+ticker+'/'
r1 = requests.get(url)
coverpage = r1.content
soup1 = BeautifulSoup(coverpage, 'html5lib')
coverpage_news = soup1.find('div', class_="lh-medium col-md-10 col-lg-9")
sections=coverpage_news.find_all('section')
for section in sections:
    #print(section.h4.text) ALL QUESTIONS YOU CAN HAVE ANSWERED
    #print(section.div.text)ALL ANSWERS TO THE QUESTIONS 
    if 'been receiving favorable news coverage?' in (section.h4.text):
        market_beat=(section.div.text)
'''

    

print('{} articles were analyzed.'.format(text_counter))
print('\n')
print("According to the "+cla+" , the probability that your review was negative was {}%,the probability it was positive was {}%, and the probability it was neutral was {}% .".format(neg_sent,pos_sent,neu_sent))
print('\n')
print("According to the"+cl+" Classifier, the probability that your review was negative was {}% and the probability it was positive was {}%.".format(n_sent,p_sent))
print('\n')
print('ARTICLE SUMMARY WITH A SENTIMENT NEAREST TO THE AVERAGE SENTIMENT')
print('\n')
print(avg_summary)
print('\n')
#print('ADDITIONAL INFO')
#print(market_beat)





#NEXT STEPS
#make text blob work with any classifier 'https://textblob.readthedocs.io/en/dev/advanced_usage.html'
##IMPLEMENT A it-idf/LDA/n-Grams test that spits out a n-gram for combined articles
#also analyze the titles and provide the summary of with a sentiment percent that most closly matches the average sentiment
#implement machine learning to imporve the naive bayes function
#FIND BETTER DATA SET TO TRAIN CLASSIFIER ON
#use pickle
#add robin hood web scraper
#find search history thats usable and minute by minute

