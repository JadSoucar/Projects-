from time import sleep
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime
import requests
from requests import get
from bs4 import BeautifulSoup
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import threading
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
from itertools import count
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d



ticker='MOM'      #STOCK TICKER
refresh_time=10.0 #IN SECONDS


values_to_plot=[]
dates=[]
line1=[]
    
def price(ticker):
    quote=0
    url=("https://finance.yahoo.com/quote/"+ticker+'?p='+ticker+'&.tsrc=fin-srch')
    response=requests.get(url)
    soup=BeautifulSoup(response.text,'lxml')
    price= soup.find_all('div',{'class':"My(6px) Pos(r) smartphone_Mt(6px)"})[0].find('span').text
    return price

def livelist():
    #if int(datetime.now().strftime('%H'))<=13:
    threading.Timer(refresh_time, livelist).start()
    values_to_plot.append(float(price(ticker)))
    dates.append(datetime.now().strftime("%M:%S"))
    return(dates),(values_to_plot)


plt.style.use('ggplot')
x_values = []
y_values = []
counter = 0
index = count()

def animate(i):
    counter =next(index)
    x_values.append((datetime.now().strftime("%H %M:%S")))
    y_values.append((price(ticker)))
    if counter >20:
        x_values.pop(0)
        y_values.pop(0)
        counter = 0
        plt.cla()
    plt.cla() # clears the values of the graph  
    plt.plot(x_values, y_values,linestyle='--',label=ticker)

    plt.legend()
    plt.xlabel("Hours Min:Sec")
    plt.ylabel("Price")
    plt.title('Live Price')
    plt.xticks(fontsize=3)
    time.sleep(.25) # keep refresh rate of 0.25 seconds

def run_animation():
    ani = FuncAnimation(plt.gcf(), animate, 1000)
    plt.tight_layout()
    plt.show()


run_animation()







