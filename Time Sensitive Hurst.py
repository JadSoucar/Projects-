import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from decimal import Decimal
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
import csv
import seaborn as sb
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
import warnings
from datetime import datetime
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
import math
from itertools import islice, chain, repeat
from numpy import cumsum, log, polyfit, sqrt, std, subtract
import numpy as np
from scipy.stats import linregress
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from hurst import compute_Hc, random_walk
import scipy.stats as stats
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
  

class stockSetup:
    time_period = ""
    data_source = ""
    def __init__(self, timePeriod, dataSource): # we could make different classes for each data acquisition type - yahoo finance, robinhood, quandl, etc
        self.time_period = timePeriod
        self.data_source = dataSource
    


class stock(stockSetup): # ticker, time_period, data_source, data, dates, values_open, values_close, percent_change
    ticker = ""
    #data
    def __init__(self, stock_ticker_symbol):
        self.ticker = stock_ticker_symbol
        self.time_period = time_period #self.getTimePeriod() 
        self.data_source = data_source
        if (data_source == "yahoo" or data_source == "yh"):
            self.data = yf.Ticker(self.ticker).history(period=time_period)
            self.dates = self.data.index.strftime("%m/%d/%Y").tolist() #List of dates in time_period
            self.values_open = self.data['Open'].tolist() # Open, Close, or High
            self.values_close = self.data['Close'].tolist()
            self.percent_change = []
            for i in range(len(self.values_close)): #could be open or close, doesn't matter
                percent_change_calc = ( ( (self.values_close[i]-self.values_close[0]) / self.values_close[0] ) * 100 ) # ((new-old)/old)*100
                self.percent_change.append(percent_change_calc)    
#        else if (data_source == "quandl" or data_source == "q"):
#            self.data = ?
#        else if (data_source == "robinhood" or data_source == "rh"):
#            self.data = ?
        else:
            print("Invalid data source. Choices are robinhood/rh, yahoo/yh, quandl/q")
    
    def getPercentChange(self):
        return self.percent_change
    def getDates(self):
        return self.dates
    def getYears(self):
        self.years = self.data.index.strftime("%Y").tolist()
        self.years = list(dict.fromkeys(self.years))
        return self.years
    def getData(self):
        return self.data
    def getValuesOpen(self):
        return self.values_open
    def getValuesClose(self):
        return self.values_close
    def __str__(self):
        return self.ticker


    
time_period = "5y"
data_source = "yahoo" #yahoo, quandl, robinhood, or shortened yh, rh, q

setup = stockSetup(time_period, data_source)

msft = stock("MSFT")
fb = stock("AAPL")

stocks = [msft, fb]
method='percent change' #percent change or price






maximum_chunk=8 #minimum_chunk doesnt matter unless maximum_chunk is under 100, this is pretty much your time scale
minimum_chunk=2 #pretty much how many lags do you want (the lower this number the more lags youll get)
r=9 #to what decimal do you want the data rounded to


def hurst_graph(stocks,maximum_chunk,minimum_chunk,r):
    def hurst(data_set,minimum_chunk,r):
        def chunk_calc(data_set):
            for num in range(len(data_set)):
                data_set_adjusted= data_set[num:]
                n=(math.log(len(data_set_adjusted),minimum_chunk))
                if (n%int(n))==0.0:
                    return int(n), data_set_adjusted
                    break

        def chunk_pad(it, size, padval=None):
            it = chain(iter(it), repeat(padval))
            return list(iter(lambda: tuple(islice(it, size)), (padval,) * size))

        expo,data_set_a=chunk_calc(data_set)


        def chunks(data_set_a):
            def first_chunk(data_set_a):
                mean=round(np.mean(data_set),r)
                sd=round(std(data_set),r)
                mean_centered_series=[]
                cumulative_deviation=[]
                for num in data_set_a:
                    mean_centered_series.append(round(num-mean,r))
                for num in range(len(mean_centered_series)):
                    cumulative_deviation.append(round(sum(mean_centered_series[:num]),r))
                Range=round(max(cumulative_deviation)-min(cumulative_deviation),r)
                rescaled_range=round((Range/sd),r)
                log_of_rs=round(log(rescaled_range),r)
                log_of_size=round(log(len(data_set_a)),r)
                return log_of_rs, log_of_size
        
            list_of_log_of_rs=[]
            list_of_log_of_size=[]
            list_of_log_of_rs.append(first_chunk(data_set_a)[0])
            list_of_log_of_size.append(first_chunk(data_set_a)[1])
            for num in (list(range(expo+1))[1:-1]):
                Ranges=[]
                rescaled_ranges=[]
                denominater=minimum_chunk**num
                num_of_chunks=(int(int(len(data_set_a))/denominater))
                for chunk in (chunk_pad(data_set_a,num_of_chunks)):
                    mean=round(np.mean(chunk),r)
                    sd=round(std(chunk),r)
                    mean_centered_series=[]
                    cumulative_deviation=[]
                    for n in chunk:
                        mean_centered_series.append(round(n-mean,r))
                    for n in range(len(chunk)):
                        cumulative_deviation.append(round(sum(mean_centered_series[:n]),r))
                    Range=round(max(cumulative_deviation)-min(cumulative_deviation),r)
                    rescaled_ranges.append(round(Range/sd,r))
                avg_rescaled_range=round(sum(rescaled_ranges)/denominater,r)
                list_of_log_of_rs.append(round(log(avg_rescaled_range),r))
                list_of_log_of_size.append(round(log(len((chunk_pad(data_set_a,num_of_chunks))[0])),r))
            #plt.scatter(list_of_log_of_size,list_of_log_of_rs)
            #plt.show()
            #return list_of_log_of_rs,list_of_log_of_size
            def invertList(input_list): 
                for item in range(len(input_list)//2): 
                    input_list[item], input_list[len(input_list)-1-item] = input_list[len(input_list)-1-item], input_list[item]
                return input_list
            Y=invertList(list_of_log_of_rs)
            X=invertList(list_of_log_of_size)
            def best_fit(X, Y):
                xbar = sum(X)/len(X)
                ybar = sum(Y)/len(Y)
                n = len(X) # or len(Y)
                numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
                denum = sum([xi**2 for xi in X]) - n * xbar**2
                b = numer / denum
                a = ybar - b * xbar
                return a
            return best_fit(X,Y)
        return np.absolute(chunks(data_set_a))

    def hurst_alt(ts):
        H, c, val = compute_Hc(ts)
        return H


    hurst_values=[]
    time_adjusted_dates=[]
    counter1=0
    counter2=maximum_chunk
    skipped=0
    for period in list(range(0,len(stocks[0].getValuesClose()),maximum_chunk))[1:]:
        difference=[]
        for i in range(counter1,counter2):
            difference.append(stocks[0].percent_change[i]- stocks[1].percent_change[i])
        time_adjusted_dates.append(stocks[0].getDates()[period])
        if maximum_chunk<100:
            hurst_values.append(hurst(difference,minimum_chunk,r))
        else:
            hurst_values.append(hurst_alt(difference))
        counter1+=maximum_chunk
        counter2+=maximum_chunk

    plt.plot(time_adjusted_dates, hurst_values)
    plt.xlabel('Dates')
    plt.ylabel('Hurst Values')
    plt.title(str(maximum_chunk)+'-Day Hurst Chart')

    #dealing with the x-axis labels    
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    years_fmt = mdates.DateFormatter('%Y')
    plt.gca().xaxis.set_major_locator(years)
    plt.gca().xaxis.set_major_formatter(years_fmt)
    plt.gca().xaxis.set_minor_locator(months)
    plt.gcf().autofmt_xdate()
        
    plt.show()
    
def t_distribution(residuals,std_threshold):
    #std_threshold can be 0,1,2,3 standard deviations  
    residuals.sort()
    residuals_mean = np.mean(residuals)
    residuals_std = np.std(residuals)
    pdf = stats.norm.pdf(residuals, residuals_mean, residuals_std)
    plt.plot(residuals, pdf) # including h here is crucial
    #SET STD THRESHOLD
    plt.axvline(residuals_mean-(std_threshold*residuals_std),color='red')
    plt.axvline(residuals_mean+(std_threshold*residuals_std),color='red')
    plt.show()
    #list of days where threshold was passed 
    alert_dic={}
    for i in range(len(residuals)):
        if residuals[i]>=(residuals_mean+(std_threshold*residuals_std)) or residuals[i]<=(residuals_mean-(std_threshold*residuals_std)):
            alert_dic[stocks[0].getDates()[i]]=residuals[i]
    
    #ACCESS ALERT DATES AND RESIDUALS
    #print(alert_dic) 
    #ACCESS ALERT DATES
    #print(alert_dic.keys())





time_period = "5y"
data_source = "yahoo" #yahoo, quandl, robinhood, or shortened yh, rh, q

setup = stockSetup(time_period, data_source)

msft = stock("MSFT")
fb = stock("AAPL")

stocks = [msft, fb]
method='percent change' #percent change or price

maximum_chunk=12 #minimum_chunk doesnt matter unless maximum_chunk is under 100, this is pretty much your time scale
minimum_chunk=2 #pretty much how many lags do you want (the lower this number the more lags youll get)
r=9 #to what decimal do you want the data rounded to


hurst_graph(stocks,maximum_chunk,minimum_chunk,r)

#t_distribution(hurst_graph(stocks,maximum_chunk,minimum_chunk,r),2)



























