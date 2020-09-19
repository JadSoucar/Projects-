import requests
from bs4 import BeautifulSoup
import random
import pandas 
import pandas_datareader.data as web
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from numpy import cumsum, log, polyfit, sqrt, std, subtract

today= datetime.datetime.today()
start_ = today - timedelta(days=365*4)
start_stock = today - timedelta(days=365)
disposable_income=web.DataReader('DSPIC96',"fred",start=start_)
personal_consumption=web.DataReader('PCEC96',"fred",start=start_)
nonfarm_payrolls=web.DataReader('PAYEMS',"fred",start=start_)
consumer_price_index=web.DataReader('CPIAUCSL',"fred",start=start_)
sp_500_uncleaned=web.DataReader('SPY',"yahoo",start=start_stock)
sp_500=sp_500_uncleaned['Adj Close']
def norm_dist(dataframe):
    listt_unindexed=[dataframe.values.tolist()[x][0] for x in range(len(dataframe))]
    listt=[x/max(listt_unindexed) for x in listt_unindexed]
    data_sorted=sorted(listt)
    data_mean = np.mean(listt)
    data_std = np.std(listt)
    if listt[-1]>=data_mean:
        z=(listt[-1]-data_mean)/data_std
    else:
        z=(data_mean-listt[-1])/data_std
    return z
def norm_dist_yahoo(dataframe):
    listt_unindexed=[dataframe.iloc[x] for x in range(len(dataframe))]
    listt=[x/max(listt_unindexed) for x in listt_unindexed]
    data_sorted=sorted(listt)
    data_mean = np.mean(listt)
    data_std = np.std(listt)
    if np.average(listt[-31:])>=data_mean:
        z=(np.average(listt[-31:])-data_mean)/data_std
    else:
        z=(data_mean-(np.average(listt[-31:]))/data_std)
    return z
z_score=((norm_dist_yahoo(sp_500)+norm_dist(disposable_income)+norm_dist(personal_consumption)+norm_dist(nonfarm_payrolls)+norm_dist(consumer_price_index))/5)


states = {
        ('AK',3): 'Alaska',
        ('AL',9): 'Alabama',
        ('AR',6): 'Arkansas',
        ('AZ',11): 'Arizona',
        ('CA',55): 'California',
        ('CO',9): 'Colorado',
        ('CT',7): 'Connecticut',
        ('DC',3): 'District of Columbia',
        ('DE',3): 'Delaware',
        ('FL',29): 'Florida',
        ('GA',16): 'Georgia',
        ('HI',4): 'Hawaii',
        ('IA',6): 'Iowa',
        ('ID',4): 'Idaho',
        ('IL',20): 'Illinois',
        ('IN',11): 'Indiana',
        ('KS',6): 'Kansas',
        ('KY',8): 'Kentucky',
        ('LA',8): 'Louisiana',
        ('MA',11): 'Massachusetts',
        ('MD',10): 'Maryland',
        ('ME',4): 'Maine',
        ('MI',16): 'Michigan',
        ('MN',10): 'Minnesota',
        ('MO',10): 'Missouri',
        ('MS',6): 'Mississippi',
        ('MT',3): 'Montana',
        ('NC',15): 'North Carolina',
        ('ND',3): 'North Dakota',
        ('NE',5): 'Nebraska',
        ('NH',4): 'New Hampshire',
        ('NJ',14): 'New Jersey',
        ('NM',5): 'New Mexico',
        ('NV',6): 'Nevada',
        ('NY',29): 'New York',
        ('OH',18): 'Ohio',
        ('OK',7): 'Oklahoma',
        ('OR',): 'Oregon',
        ('PA',20): 'Pennsylvania',
        ('RI',4): 'Rhode Island',
        ('SC',9): 'South Carolina',
        ('SD',3): 'South Dakota',
        ('TN',11): 'Tennessee',
        ('TX',38): 'Texas',
        ('UT',6): 'Utah',
        ('VA',13): 'Virginia',
        ('VT',3): 'Vermont',
        ('WA',12): 'Washington',
        ('WI',10): 'Wisconsin',
        ('WV',5): 'West Virginia',
        ('WY',3): 'Wyoming'
}


red_states={('ID', 4): 'Idaho',('ND', 3):'North Dakota',
            ('NE', 5): 'Nebraska', ('OK', 7): 'Oklahoma',
            ('SD', 3): 'South Dakota',('WV', 5): 'West Virginia',
            ('WY', 3): 'Wyoming'}


blue_states={('DC', 3): 'District of Columbia',('IL', 20): 'Illinois',
             ('OR',7): 'Oregon',('RI', 4): 'Rhode Island',('VT', 3):
             'Vermont'}


links={
    'Wisconsin':'https://www.realclearpolitics.com/epolls/2020/president/wi/wisconsin_trump_vs_biden-6849.html',
    'Florida':'https://www.realclearpolitics.com/epolls/2020/president/fl/florida_trump_vs_biden-6841.html',
    'Pennsylvania': 'https://www.realclearpolitics.com/epolls/2020/president/pa/pennsylvania_trump_vs_biden-6861.html',
    'North Carolina':'https://www.realclearpolitics.com/epolls/2020/president/nc/north_carolina_trump_vs_biden-6744.html',
    'Michigan':'https://www.realclearpolitics.com/epolls/2020/president/mi/michigan_trump_vs_biden-6761.html',
    'Arizona': 'https://www.realclearpolitics.com/epolls/2020/president/az/arizona_trump_vs_biden-6807.html',
    'Minnesota':'https://www.realclearpolitics.com/epolls/2020/president/mn/minnesota_trump_vs_biden-6966.html',
    'Ohio':'https://www.realclearpolitics.com/epolls/2020/president/oh/ohio_trump_vs_biden-6765.html',
    'Iowa':'https://www.realclearpolitics.com/epolls/2020/president/ia/iowa_trump_vs_biden-6787.html',
    'Nevada':'https://www.realclearpolitics.com/epolls/2020/president/nv/nevada_trump_vs_biden-6867.html',
    'New Hampshire':'https://www.realclearpolitics.com/epolls/2020/president/nh/new_hampshire_trump_vs_biden-6779.html',
    'Maine':'https://www.realclearpolitics.com/epolls/2020/president/me/maine_trump_vs_biden-6922.html',
    'Virginia':'https://www.realclearpolitics.com/epolls/2020/president/va/virginia_trump_vs_biden-6988.html',
    'Georgia':'https://www.realclearpolitics.com/epolls/2020/president/ga/georgia_trump_vs_biden-6974.html',
    'Texas':'https://www.realclearpolitics.com/epolls/2020/president/tx/texas_trump_vs_biden-6818.html',
    'Colorado':'https://www.realclearpolitics.com/epolls/2020/president/co/colorado_trump_vs_biden-6940.html',
    'New Mexico':'https://www.realclearpolitics.com/epolls/2020/president/nm/new_mexico_trump_vs_biden-6993.html',
    "Alaska": 'https://www.realclearpolitics.com/epolls/2020/president/ak/alaska_trump_vs_biden-7219.html',
    "Alabama": 'https://www.realclearpolitics.com/epolls/2020/president/al/alabama_trump_vs_biden-7022.html',
    "Arkansas":'https://www.realclearpolitics.com/epolls/2020/president/ar/arkansas_trump_vs_biden-7213.html',
    "Kansas":'https://www.realclearpolitics.com/epolls/2020/president/ks/kansas_trump_vs_biden-7058.html',
    "Kentucky":'https://www.realclearpolitics.com/epolls/2020/president/ky/kentucky_trump_vs_biden-6915.html',
    "Louisiana":'https://www.realclearpolitics.com/epolls/2020/president/la/louisiana_trump_vs_biden-7245.html',
    "Missouri":'https://www.realclearpolitics.com/epolls/2020/president/mo/missouri_trump_vs_biden-7210.html',
    "Mississippi":'https://www.realclearpolitics.com/epolls/2020/president/ms/mississippi_trump_vs_biden-7052.html',
    "Montana":'https://www.realclearpolitics.com/epolls/2020/president/mt/montana_trump_vs_biden-7196.html',
    "South Carolina":'https://www.realclearpolitics.com/epolls/2020/president/sc/south_carolina_trump_vs_biden-6825.html',
    "Tennessee":'https://www.realclearpolitics.com/epolls/2020/president/tn/tennessee_trump_vs_biden-7006.html',
    "Utah":'https://www.realclearpolitics.com/epolls/2020/president/ut/utah_trump_vs_biden-7195.html',
    "California":'https://www.realclearpolitics.com/epolls/2020/president/ca/california_trump_vs_biden-6755.html',
    "Connecticut":'https://www.realclearpolitics.com/epolls/2020/president/ct/connecticut_trump_vs_biden-6999.html',
    'Delaware':'https://www.realclearpolitics.com/epolls/2020/president/de/delaware_trump_vs_biden-7028.html',
    'Hawaii':'https://www.realclearpolitics.com/epolls/2020/president/hi/hawaii_trump_vs_biden-7233.html',
    'Indiana':'https://www.realclearpolitics.com/epolls/2020/president/in/indiana_trump_vs_biden-7189.html',
    'Massachusetts':'https://www.realclearpolitics.com/epolls/2020/president/ma/massachusetts_trump_vs_biden-6876.html',
    'Maryland':'https://www.realclearpolitics.com/epolls/2020/president/md/maryland_trump_vs_biden-7209.html',
    'New Jersey':'https://www.realclearpolitics.com/epolls/2020/president/nj/new_jersey_trump_vs_biden-7193.html',
    'Nevada':'https://www.realclearpolitics.com/epolls/2020/president/nv/nevada_trump_vs_biden-6867.html',
    'New York':'https://www.realclearpolitics.com/epolls/2020/president/ny/new_york_trump_vs_biden-7040.html',
    'Washington':'https://www.realclearpolitics.com/epolls/2020/president/wa/washington_trump_vs_biden-7060.html'
}

bd=1*float(70/294)
td=1*float(z_score)
#bias, the closer to 0 and negative the more positive bias 


polling={}
for link in links.items():
    poles=[]
    url=link[1]
    r1 = requests.get(url)
    coverpage = r1.content
    soup1 = BeautifulSoup(coverpage, 'html5lib')
    coverpage_news = soup1.find_all('div', class_="polling_dt layout_0")
    for item in coverpage_news:
        info=(item.table.tbody.children)
        break
    for item in info:
        moe=0
        biden=0
        trump=0
        if '--' not in(item.text) and 'PollDateSample' not in item.text:
            if 'LV' in item.text:
                moe=((item.text.split('LV')[1])[0:3])
                biden=((item.text.split('LV')[1])[3:5])
                trump=((item.text.split('LV')[1])[5:7])
                poles.append([float(moe),int(biden),int(trump)])
   
            elif 'RV' in item.text:
                moe=((item.text.split('RV')[1])[0:3])
                biden=((item.text.split('RV')[1])[3:5])
                trump=((item.text.split('RV')[1])[5:7])
                poles.append([float(moe),int(biden),int(trump)])
    polling[link[0]]=poles


biden_wins=0
trump_wins=0
for x in range(100):
    winner={}
    for state in polling.items():
        trump_s=0
        biden_s=0
        for pole in state[1]:
            biden=(random.randint(round(pole[1]-pole[0]*(1/bd)), round(pole[1]+pole[0]*bd)))
            trump=(random.randint(round(pole[2]-pole[0]*(1/td)), round(pole[2]+pole[0]*td)))
            if trump>biden:
                trump_s+=1
            elif trump<biden:
                biden_s+=1
            else:
                continue

        if trump_s>biden_s:
            winner[state[0]]='Trump'
        elif biden_s>trump_s:
            winner[state[0]]='Biden'
        else:
            for x in range(100):
                trump_sum=0
                biden_sum=0
                for pole in state[1]:
                    biden1=(random.randint(round(pole[1]-pole[0]*(1/bd)), round(pole[1]+pole[0]*bd)))
                    trump1=(random.randint(round(pole[2]-pole[0]*(1/td)), round(pole[2]+pole[0]*td)))
                    if trump1>biden1:
                        trump_sum+=1
                    elif trump1<biden1:
                        biden_sum+=1
                    else:
                        continue
                if trump_sum>biden_sum:
                    winner[state[0]]='Trump'
                    break
                elif biden_sum>trump_sum:
                    winner[state[0]]='Biden'
                    break
                else:
                    continue 

    biden_elect=0
    trump_elect=0
    states_new={value:key for key, value in states.items()}

    for win in winner.items():
        if win[1]=='Biden':
            biden_elect+=states_new[win[0]][1]
        elif win[1]=='Trump':
            trump_elect+=states_new[win[0]][1]
        else:
            print('ERROR')

    for state in red_states.items():
        trump_elect+=int(state[0][1])
    for state in blue_states.items():
        biden_elect+=int(state[0][1])

    if biden_elect>trump_elect:
        biden_wins+=1
    else:
        trump_wins+=1

    print("Trump: "+ str(trump_elect))
    print("Biden: "+ str(biden_elect))
    print('\n')
    

print('##########################################')
print("Trump: "+ str(trump_wins))
print("Biden: "+ str(biden_wins))
print('\n')



#turnout model
#variability model (model based on headlines, and congressional votes)
#coronavirus
