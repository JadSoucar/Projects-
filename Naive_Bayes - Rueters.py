from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk, re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from text_preprocessing import preprocess_text
from text_preprocessing import to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word
from nltk.corpus import reuters
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
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import classifiers


###########################GET ARTICLES########################################
ticker='AAPL'
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

    
#########################PREPARE REUTERS CORPUS AND TRAIN CLASSIFIER######################################################
preprocessed_corpus=[]

for fid in reuters.fileids():
    preprocessed_corpus.append(preprocess_text(reuters.words(fid)))
        
cleaned_preprocessed_corpus=[]


# creating the bag of words model
bag_of_words_creator = CountVectorizer()
bag_of_words = bag_of_words_creator.fit_transform(cleaned_preprocessed_corpus)

# creating the tf-idf model
tfidf_creator = TfidfVectorizer(min_df = 0.2)
tfidf = tfidf_creator.fit_transform(preprocessed_corpus)



documents=[(list(reuters.words(fileid)), category)
           for category in reuters.categories()
           for fileid in reuters.fileids(category)]
random.shuffle(documents)


word_features=(tfidf_creator.get_feature_names()[10000])


def find_features(document):
    words=set(document)
    features={}
    for w in word_features:
        features[w]=(w in words)
    return features

featursets=[(find_features(rev),category) for (rev,category) in documents]
training_set=featursets[:2000]
testing_set=featursets[2000:]

classifier = classifiers.NaiveBayesClassifier(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier = open("naivebayes_reuters.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()



#################RUN TEXT THROUGH CLASSIFIER##############################
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



#######################PRINT RESULTS######################################
print("According to the"+cl+" Classifier, the probability that your review was negative was {}% and the probability it was positive was {}%.".format(n_sent,p_sent))
print('\n')
print('ARTICLE SUMMARY WITH A SENTIMENT NEAREST TO THE AVERAGE SENTIMENT')
print('\n')
print(avg_summary)



#pickle it
#100 top key words form the newspaper module
#see if i can make it work with other classifiers
#find minute by minute search history and set up system to test whenever searches spike 
