import pandas_datareader.data as web
import datetime
import requests
import requests
from requests import get
from bs4 import BeautifulSoup
from datetime import date
from decimal import Decimal
import retry as retry



ticker='ET'
years=5
method='levered' #options are levered, unlevered, EBITA
tv_method='exit' #option are terminal are exit









def fcf(ticker,method):
    if method=='levered':
        levered=[]
        url=('https://finance.yahoo.com/quote/'+ticker+'/cash-flow?p='+ticker)
        response=requests.get(url)
        soup=BeautifulSoup(response.text,'html.parser')
        net_flow=soup.find_all('div', class_="Ta(c) Py(6px) Bxz(bb) BdB Bdc($seperatorColor) Miw(120px) Miw(140px)--pnclg Bgc($lv1BgColor) fi-row:h_Bgc($hoverBgColor) D(tbc)")
        for container in net_flow:
            levered.append(container.find('span'))
        strr=(str(levered).split('>,'))[-3]
        return (float((strr[(strr.find('>')+1):strr.find('</')]).replace(',',''))*1000)
    if method=='unlevered':
        unlevered=0
        url=('https://finance.yahoo.com/quote/'+ticker+'/cash-flow?p='+ticker)
        response=requests.get(url)
        soup=BeautifulSoup(response.text,'html.parser')
        net_flow=soup.find_all('div', class_="Ta(c) Py(6px) Bxz(bb) BdB Bdc($seperatorColor) Miw(120px) Miw(140px)--pnclg Bgc($lv1BgColor) fi-row:h_Bgc($hoverBgColor) D(tbc)")
        for container in net_flow:
            unlevered=(container.find('span').text)
            break
        unlevered_1=unlevered.replace(',','')
        return float(unlevered_1)*1000
    if method=='EBITA':
        ebit=[]
        ebit_num=0
        url='https://www.macrotrends.net/stocks/charts/'+ticker+'/apple/ebitda'
        response=requests.get(url)
        soup=BeautifulSoup(response.text,'html.parser')
        ebitl=(soup.find_all('div',style='background-color:#fff; margin: 0px 0px 20px 0px; padding:20px 30px; border:1px solid #dfdfdf;'))
        for container in ebitl:
            ebit=container.find('ul', style="margin-top:10px;").text
        real_ebit=((ebit.split('\n'))[2])
        legit_ebit=(real_ebit[real_ebit.find('$'):])
        if 'B' in legit_ebit:
            ebit_num=(float(legit_ebit[legit_ebit.find('$')+1:legit_ebit.find('B')]))*1000000000
        if 'M' in legit_ebit:
            ebit_num=(float(legit_ebit[legit_ebit.find('$')+1:legit_ebit.find('B')]))*1000000
        return ebit_num




def EBIT(ticker):
    ebit=[]
    ebit_num=0
    url=('https://www.macrotrends.net/stocks/charts/'+ticker+'/apple/ebit?q='+ticker)
    response=requests.get(url)
    soup=BeautifulSoup(response.text,'html.parser')
    ebitl=(soup.find_all('div',style='background-color:#fff; margin: 0px 0px 20px 0px; padding:20px 30px; border:1px solid #dfdfdf;'))
    for container in ebitl:
        ebit=container.find('ul', style="margin-top:10px;").text
    real_ebit=((ebit.split('\n'))[2])
    a_ebit=real_ebit.split(' ')
    legit_ebit=a_ebit[11].strip('$').strip(',')
    if legit_ebit[-1]=='B':
        ebit_num=(float(legit_ebit.strip('B')))*1000000000
    if legit_ebit[-1]=='M':
        ebit_num=(float(legit_ebit.strip('M')))*1000000
    return ebit_num


def tax_rate():
    rate=0
    url='https://tradingeconomics.com/united-states/corporate-tax-rate'
    response=requests.get(url)
    soup=BeautifulSoup(response.text,'html.parser')
    tax_rate=soup.find_all('td')
    rate=str(tax_rate[2])
    return Decimal('.'+((rate.strip('<td>').strip('</td>'))[0:2]))



  
def interest(ticker):
    num_1=[]
    url='https://www.gurufocus.com/term/InterestExpense/'+ticker+'/Interest-Expense/Apple-Inc'
    response=requests.get(url)
    soup=BeautifulSoup(response.text,'html.parser')
    coverage=soup.find_all('div', class_="col-lg-10 col-md-10 col-xs-12")
    strr=str(coverage)
    strr_cleaned=(strr[strr.find('Interest Expense'):strr.find('(TTM As of')])
    final=strr_cleaned[strr_cleaned.find('>:')+5:]
    return ((Decimal(float(final[:-4].replace(',',''))))*Decimal(1000000))


def Beta(ticker):
    beta=[]
    url=('https://finance.yahoo.com/quote/'+ticker)
    response=requests.get(url)
    soup=BeautifulSoup(response.text,'html.parser')
    price=(soup.find_all('td',class_="Ta(end) Fw(600) Lh(14px)"))
    strr=(str(price).split('"'))
    strr_short=strr[92]
    num=(strr_short[(strr_short.find('>')+1):strr_short.find('<')])
    result =Decimal(num)
    return result


def total_debt(ticker):
    debt=0
    total_debt=[]
    EV=0
    url='https://www.macroaxis.com/invest/financial-statements/'+ticker+'/Total-Debt'
    response=requests.get(url)
    soup=BeautifulSoup(response.text,'html.parser')
    price=(soup.find_all('span',style="margin:0;padding:0;font-size:1em"))
    for container in price:
        debt=str(container.find('strong').text)
    debt_1=debt.replace('\xa0','')
    if 'T' in debt_1:
        EV=float(debt_1.replace('T',''))*1000000000000
    if 'B' in debt_1:
        EV=float(debt_1.replace('B',''))*1000000000
    if 'M' in debt_1:
        EV=float(debt_1.replace('M',''))*1000000
    shares_outstanding=EV
    return EV

        
def total_stockholder_equity(ticker):
    num_1=0
    changes=[]
    changes_cleaned=[]
    url='https://www.macrotrends.net/stocks/charts/'+ticker+'/apple/total-share-holder-equity'
    response=requests.get(url)
    soup=BeautifulSoup(response.text,'html.parser')
    combo_data=soup.find_all('div', style="background-color:#fff; margin: 0px 0px 20px 0px; padding:20px 30px; border:1px solid #dfdfdf;")
    combo_data_1=str(combo_data).split(',')
    for change in combo_data_1:
        if 'a <strong>' and '</strong>' in change:
            changes.append(change[(change.find('a <strong>')+1):change.find('</strong>')])
        else:
            continue
    changes_cleaned=(changes[0])
    debt_1=changes_cleaned[(changes_cleaned.find('<strong>$')+9):]
    if 'T' in debt_1:
        EV=float(debt_1.replace('T',''))*1000000000000
    if 'B' in debt_1:
        EV=float(debt_1.replace('B',''))*1000000000
    if 'M' in debt_1:
        EV=float(debt_1.replace('M',''))*1000000

    return EV




###########################################WACC CALC################################################################################
def interest_coverage_and_RF(ticker):
    ebit=EBIT(ticker)
    interest_expense=interest(ticker)
    interest_coverage_ratio=Decimal(ebit)/Decimal(interest_expense)
    #RF
    end= datetime.datetime.today().strftime('%Y-%m-%d')
    start = date((datetime.datetime.today()).year -1, (datetime.datetime.today()).month, (datetime.datetime.today()).day).strftime('%Y-%m-%d')

        
    Treasury = web.DataReader(['TB1YR'], 'fred', start, end)
    RF = float(Treasury.iloc[-1])
    RF = RF/100
    return [RF,interest_coverage_ratio]


RF, interest_coverage_ratio=interest_coverage_and_RF(ticker)

#Cost of debt
def cost_of_debt(ticker, RF,interest_coverage_ratio):
  if interest_coverage_ratio > 8.5:
    #Rating is AAA
    credit_spread = 0.0063
  if (interest_coverage_ratio > 6.5) & (interest_coverage_ratio <= 8.5):
    #Rating is AA
    credit_spread = 0.0078
  if (interest_coverage_ratio > 5.5) & (interest_coverage_ratio <=  6.5):
    #Rating is A+
    credit_spread = 0.0098
  if (interest_coverage_ratio > 4.25) & (interest_coverage_ratio <=  5.49):
    #Rating is A
    credit_spread = 0.0108
  if (interest_coverage_ratio > 3) & (interest_coverage_ratio <=  4.25):
    #Rating is A-
    credit_spread = 0.0122
  if (interest_coverage_ratio > 2.5) & (interest_coverage_ratio <=  3):
    #Rating is BBB
    credit_spread = 0.0156
  if (interest_coverage_ratio > 2.25) & (interest_coverage_ratio <=  2.5):
    #Rating is BB+
    credit_spread = 0.02
  if (interest_coverage_ratio > 2) & (interest_coverage_ratio <=  2.25):
    #Rating is BB
    credit_spread = 0.0240
  if (interest_coverage_ratio > 1.75) & (interest_coverage_ratio <=  2):
    #Rating is B+
    credit_spread = 0.0351
  if (interest_coverage_ratio > 1.5) & (interest_coverage_ratio <=  1.75):
    #Rating is B
    credit_spread = 0.0421
  if (interest_coverage_ratio > 1.25) & (interest_coverage_ratio <=  1.5):
    #Rating is B-
    credit_spread = 0.0515
  if (interest_coverage_ratio > 0.8) & (interest_coverage_ratio <=  1.25):
    #Rating is CCC
    credit_spread = 0.0820
  if (interest_coverage_ratio > 0.65) & (interest_coverage_ratio <=  0.8):
    #Rating is CC
    credit_spread = 0.0864
  if (interest_coverage_ratio > 0.2) & (interest_coverage_ratio <=  0.65):
    #Rating is C
    credit_spread = 0.1134
  if interest_coverage_ratio <=  0.2:
    #Rating is D
    credit_spread = 0.1512
  
  cost_of_debt = RF + credit_spread
  return cost_of_debt

def costofequity(ticker):
    #RF
    end= datetime.datetime.today().strftime('%Y-%m-%d')
    start = date((datetime.datetime.today()).year -1, (datetime.datetime.today()).month, (datetime.datetime.today()).day).strftime('%Y-%m-%d')
    Treasury = web.DataReader(['TB1YR'], 'fred', start, end)
    RF = float(Treasury.iloc[-1])
    RF = RF/100
   #Beta
    beta=Beta(ticker)

    #Market Return
    end= datetime.datetime.today().strftime('%Y-%m-%d')
    start = date((datetime.datetime.today()).year -1, (datetime.datetime.today()).month, (datetime.datetime.today()).day).strftime('%Y-%m-%d')

    SP500 = web.DataReader(['sp500'], 'fred', start, end)
    #Drop all Not a number values using drop method.
    SP500.dropna(inplace = True)

    SP500yearlyreturn = (SP500['sp500'].iloc[-1]/ SP500['sp500'].iloc[-252])-1
    
    cost_of_equity = Decimal(RF)+(beta*Decimal(SP500yearlyreturn - RF))
    return cost_of_equity


#effective tax rate and capital structure
def wacc(ticker):
    ETR=Decimal(tax_rate())
    TD=Decimal(total_debt(ticker))
    TSE=Decimal(total_stockholder_equity(ticker))
    ke = costofequity(ticker)
    kd = Decimal(cost_of_debt(ticker,RF,interest_coverage_ratio))

    Debt_to = TD / (TD + TSE)
    equity_to = TSE / (TD + TSE)

    WACC = (kd*(1-ETR)*Debt_to) + (ke*equity_to)
    return WACC



#print('wacc of ' + ticker + ' is ' + str((wacc(ticker)*100))+'%')


########################################################START DCF#########################################################################################
  

def revenue_growth(ticker):
    num_1=0
    changes=[]
    changes_cleaned=[]
    summ=0
    url='https://www.macrotrends.net/stocks/charts/'+ticker+'/microsoft/revenue'
    response=requests.get(url)
    soup=BeautifulSoup(response.text,'html.parser')
    combo_data=soup.find_all('div', style="background-color:#fff; margin: 0px 0px 20px 0px; padding:20px 30px; border:1px solid #dfdfdf;")
    combo_data_1=str(combo_data).split(',')
    for change in combo_data_1:
        if 'a <strong>' and '</strong>' in change:
            changes.append(change[(change.find('a <strong>')+1):change.find('</strong>')])
        else:
            continue
    changes.pop(0)
    changes.pop(1)
    for chang in changes:
        changes_cleaned.append(chang[9:])
    for chang in changes_cleaned[1:]:
        if 'decline' in chang:
            summ+=(-float(chang[:chang.find('%')]))
        if 'increase' in chang:
            summ+=(float(chang[:chang.find('%')]))
        else:
            continue
    return float(summ/(len(changes_cleaned)-1))


def sum_pv_values(ticker,years):
    last_year=fcf(ticker,method)
    expected_revenues=[fcf(ticker,method)]
    pv=[]
    for i in range(years):
        expected_revenues.append(last_year*(1+(revenue_growth(ticker)*.01)))
        last_year=last_year*(1+(revenue_growth(ticker)*.01))
    for rev in expected_revenues:
        pv.append(float(rev)/((1+float(wacc(ticker)))**((expected_revenues.index(rev))+1)))
    return sum(pv)


def TV(ticker,tv_method):
    last_year=fcf(ticker,method)
    expected_revenues=[fcf(ticker,method)]
    for i in range(years):
        expected_revenues.append(last_year*(1+(revenue_growth(ticker)*.01)))
    if tv_method=='terminal':
        tv=float(expected_revenues[-1])*float((1+float(revenue_growth(ticker)*.01))/(float(wacc(ticker))-(1+(revenue_growth(ticker)*.01))))
        return tv
    if tv_method=='exit':
        ev=0
        EV=0
        url='https://ycharts.com/companies/'+ticker+'/enterprise_value'
        response=requests.get(url)
        soup=BeautifulSoup(response.text,'html.parser')
        ev_group=soup.find_all('div', id="colMainPct")
        for container in ev_group:
            ev=container.find('span').text
        ev_1=ev[:ev.find('for')]
        if 'T' in ev_1:
            EV=float(ev_1.replace('T',''))*1000000000000
        if 'B' in ev_1:
            EV=float(ev_1.replace('B',''))*1000000000
        if 'M' in ev_1:
            EV=float(ev_1.replace('M',''))*1000000
        return expected_revenues[-1]*(EV/fcf(ticker,'EBITA'))

def values(ticker):
    num_1=0
    changes=[]
    changes_cleaned=[]
    url='https://www.macrotrends.net/stocks/charts/'+ticker+'/apple/shares-outstanding'
    response=requests.get(url)
    soup=BeautifulSoup(response.text,'html.parser')
    combo_data=soup.find_all('div', style="background-color:#fff; margin: 0px 0px 20px 0px; padding:20px 30px; border:1px solid #dfdfdf;")
    combo_data_1=str(combo_data).split(',')
    for change in combo_data_1:
        if 'a <strong>' and '</strong>' in change:
            changes.append(change[(change.find('a <strong>')+1):change.find('</strong>')])
        else:
            continue
    changes_cleaned=(changes[0])
    debt_1=changes_cleaned[(changes_cleaned.find('>')+1):]
    if 'T' in debt_1:
        EV=float(debt_1.replace('T',''))*1000000000000
    if 'B' in debt_1:
        EV=float(debt_1.replace('B',''))*1000000000
    if 'M' in debt_1:
        EV=float(debt_1.replace('M',''))*1000000
    shares_outstanding=EV
    per_share=(float(sum_pv_values(ticker,years))+(float(TV(ticker,tv_method))))/float(shares_outstanding)
    return per_share

def price(ticker):
    quote=0
    url=('https://finance.yahoo.com/quote/'+ticker)
    response=requests.get(url)
    soup=BeautifulSoup(response.text,'html.parser')
    price=(soup.find_all('div',class_='D(ib) Mend(20px)'))
    for container in price:
        quote=container.find('span', class_="Trsdu(0.3s) Fw(b) Fz(36px) Mb(-4px) D(ib)").text
    return quote

def comands(ticker):
    counter=0
    while counter<=5:
        try:
            print('Intrinsic Value:'+ str(values(ticker)))
            print('Market Value:' + str(price(ticker)))
            break
        except ValueError:
            counter+=1
            continue
        except IOError:
            counter+=1
            continue
        except AssertionError:
            counter+=1
            continue
        except AttributeError:
            counter+=1
            continue

    if counter>5:
        print('my code stinks so only 3/5 stocks work, you just stubled on one that doesnt, sorry boss')
    else:
        print('Retry Count:'+str(counter))
        
    



comands(ticker)




#FIX terminal tv
#replace ycharts for shares outstanding 
#



