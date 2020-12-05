import requests
from bs4 import BeautifulSoup
from newspaper import Article
import newspaper
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

from pytrends.request import TrendReq
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_datareader.data as web
import pandas as pd
import datetime
import requests
import arrow
from itertools import count
from requests import get
import time
import threading
from DailyTrends.collect import collect_data
from hurst import compute_Hc, random_walk

#Write Summaries for each 

class EventDriven:
    def bollinger(self, ts, window='20', label='timseries'):
        '''
        ts: input a list, and that list will  be turned into a pandas dataframe
        that also includes the lower, middle, and upper boolinger band
        window: default=20, include the window for the moving average
        label: add a label for your time series
        '''

        df= pd.DataFrame(columns=[label])
        df[label]=ts
        df['MA'] = df[label].rolling(window).mean()
        df['STD'] = df[label].rolling(window).std() 
        df['Upper'] = df['MA'] + (df['20dSTD'] * 2)
        df['Lower'] = df['MA'] - (df['20dSTD'] * 2)

    def hurst(self, ts, plot=False):
        '''
        ts: input a list you want to find the hurst value of
        plot: default=False, plot the lags on a log scale, this value must be a boolean
        '''
        H, c, val = compute_Hc(ts)
        axes = plt.subplots()[1]

        if plot==True:
            axes.plot(val[0], c*val[0]**H, color="blue")
            axes.scatter(val[0], val[1], color="red")
            axes.set_xscale('log')
            axes.set_yscale('log')
            axes.set_xlabel('Time interval')
            axes.set_ylabel('R/S ratio')
            axes.grid(True)
            plt.show
            return H
        elif plot==False:
            return H
        else:
            raise ValueError('Plot must be set to True or False')

    def intra_trends(self, ticker, window=20, timeframe='now 4-H', boolinger=False):

        '''
        ticker= include the ticker symbol for the stock you want data for
        boolinger= include a boolean for whether you want your df to be
        returned with the lower, middle, and upper boolinger
        window: default=20, include the window for the moving average
        timeframe: the formating is as follows
        timeframe

        When do you want the Date to start from

        Defaults to last 5yrs, 'today 5-y'.

        Everything 'all'

        Specific dates, 'YYYY-MM-DD YYYY-MM-DD' example '2016-12-14 2017-01-25'

        Specific datetimes, 'YYYY-MM-DDTHH YYYY-MM-DDTHH' example '2017-02-06T10 2017-02-12T07'

        Note Time component is based off UTC
        Current Time Minus Time Pattern:

        By Month: 'today #-m' where # is the number of months from that date to pull data for

        For example: 'today 3-m' would get data from today to 3months ago
        NOTE Google uses UTC date as 'today'
        Seems to only work for 1, 2, 3 months only
        Daily: 'now #-d' where # is the number of days from that date to pull data for

        For example: 'now 7-d' would get data from the last week
        Seems to only work for 1, 7 days only
        Hourly: 'now #-H' where # is the number of hours from that date to pull data for

        For example: 'now 1-H' would get data from the last hour
        Seems to only work for 1, 4 hours only
        '''
        window=20   

        keyword = [ticker+' stock']
        pytrends = TrendReq(hl='en-US', tz='360', timeout=(10, 25), retries=2,backoff_factor=1)
        pytrends.build_payload(keyword, cat=7, timeframe='now 4-H', geo='', gprop='')
        df = pytrends.interest_over_time()

        df.columns = ['relevance', 'is_partial']
        df= df.drop(['is_partial'], axis=1)

        if boolinger==True:

            df['MA20'] = df['relevance'].rolling(window).mean()
            df['20dSTD'] = df['relevance'].rolling(window).std() 
            df['Upper'] = df['MA20'] + (df['20dSTD'] * 2)
            df['Lower'] = df['MA20'] - (df['20dSTD'] * 2)
            return df
        elif boolinger==False:
            return df

        else:
            raise ValueError('Boolinger must be set to True or False')

        return(df)

    def intra_quotes(self, symbol, window=20, data_range='2m', data_interval='1m', boolinger=False):
        '''
        boolinger= include a boolean for whether you want your df to be
        returned with the lower, middle, and upper boolinger
        window: default=20, include the window for the moving average
        data_range=
        valid periods:#m, #h, #d,#wk, #mo, #y
        (optional, default is '1mo')
        data_interval=
        fetch data by interval (including intraday if period < 60 days)
        valid intervals: #m,#h,#d,#wk,#mo,#y

        Example:
            data_range=1h
            data_interval=1m

            data_range=1y
            data_interval=1h

            data_range=1wk
            data_range=1m

        NOTE: If data_range is greater then 7d you cant use a data_interval of 1m
              if data_range is greater then 2y you cant use a data_interval of 1h
              You can use data_interval of 1d or greater for any data_range greater then 1d
        '''
        res = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={data_interval}'.format(**locals()))
        data = res.json()
        body = data['chart']['result'][0]
        dt = datetime.datetime
        dt = pd.Series(map(lambda x: arrow.get(x).datetime.replace(tzinfo=None), body['timestamp']), name='Datetime')
        df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
        dg = pd.DataFrame(body['timestamp'])
        df = df.loc[:, ('open', 'high', 'low', 'close', 'volume')]
        if boolinger==True:

            df['MA20'] = df['open'].rolling(window).mean()
            df['20dSTD'] = df['open'].rolling(window).std() 
            df['Upper'] = df['MA20'] + (df['20dSTD'] * 2)
            df['Lower'] = df['MA20'] - (df['20dSTD'] * 2)
            return df
        elif boolinger==False:
            return df

        else:
            raise ValueError('Boolinger must be set to True or False')

        return(df)

    def sentiment( self, ticker, number_of_articles=50, text_boolean=False):

        '''
        classifier: default='nb'. Choose between a Niave Bayes Classifier (input='nb'), or NLTK's Sentiment
        Intensity Analyzer (input='si'). The method scrapes the top "specified number of articles"
        from google news. The classifier analyzes each article and averages the negative and positive
        scores to return a dictionary of scores. {'pos': x, 'neg': y}.
        ticker:choose the ticker symbol you want analyzed
        number_of_articles: Default= 50. Choose the number of articles you want scraped from google news, 
        text_boolean:Default=False. input True or False, if you want the text of the article that is closset to the average
        positive and negative score return in the dictionary. If this argument is 'True', the dictionary will have three keys
        {'pos': x, 'neg': y, 'text': summary}
        '''
        classifier='nb'
        ticker= ticker
        articles_examined= number_of_articles
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
            print(link)
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

        if classifier=='nb':
            import pickle
            classifier_f = open("naivebayes.pickle", "rb")
            classifier = pickle.load(classifier_f)
            classifier_f.close()

            text_counter=0
            texts_neg_sum=[]
            texts_pos_sum=[]
            result_te=''
            for text in texts:
                print('text')
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

            if text_boolean==True:
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

                return {'pos':p_sent, 'neg':n_sent, 'text':avg_summary}

            elif text_boolean==False:
                return {'pos':p_sent, 'neg':n_sent}
            

            else:
                raise ValueError('text_boolean must be either True or False')

        else:
            raise ValueError('Argument must be nb(Naive Bayes Classifier) or si(Sentiemnt Intensity Classifier)')
            


    
