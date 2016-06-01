from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import sys
import traceback
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics, preprocessing, cross_validation, svm
import seaborn as sns
import matplotlib as mlp
import matplotlib.pyplot as plt
import math

bucket = {}
table_header = []
table_data = {}
dt = []
tb_data = []
tb_d = []
tr_data = []

def get_data(stock):
    stock = stock.lower()
    stock_data = 'http://finance.yahoo.com/q/hp?s=' + stock + '&a=04&b=10&c=2011&d=04&e=5&f=2016&g=m'
    cmc = requests.get(stock_data)
    bucket['content'] = cmc.content

def parse_data(content):
    soup = BeautifulSoup(content, 'lxml')
    for th in soup.find_all('th', class_='yfnc_tablehead1'):
        table_header.append(str(th.get_text()).strip())

    for row in soup.find_all('tr'):
        for value in row.find_all('td', class_='yfnc_tabledata1'):
            if value.has_attr('nowrap'):
                new_list = []
                table_data[str(value.get_text())] = str(value.get_text())
                # table_data['Date'] = str(value.get_text())
                # date = value.get_text()
                # print date
                first = value.findNext('td')
                second = first.findNext('td')
                third = second.findNext('td')
                fourth = third.findNext('td')
                fifth = fourth.findNext('td')
                sixth = fifth.findNext('td')
                # new_list.append(str(date))
                new_list.append(str(first.get_text()))
                new_list.append(str(second.get_text()))
                new_list.append(str(third.get_text()))
                new_list.append(str(fourth.get_text()))
                new_list.append(str(fifth.get_text()).replace(',', ''))
                new_list.append(str(sixth.get_text()))
                table_data[str(value.get_text())] = new_list
                # print new_list


def data_cleansing(df):
    df = df
    df.set_index(0)
    df.index.names = ['Date']
    # print df
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'index2']
    df2 = df[['Open','Close', 'Volume']]
    df3 = df2.apply(pd.to_numeric, errors='coerce')
    df3['Mean Price'] = (df3['Open'] + df3['Close'])/2
    df3['%_change'] = (df3['Close'] - df3['Open']) / df3['Open'] * 100
    df3['index2'] = df['index2']
    # print df3
    df3.index = pd.to_datetime(df3.index)
    df5 = df3.sort()
    # print df5
    return df5

def linear_model(df):
    dff = df
    dff['index2'] = pd.to_datetime(dff['index2'])
    df2 = dff.fillna(0)
    df2 = df2[pd.notnull(df2[['Mean Price', 'Volume', '%_change', 'index2']])]
    df2 = df2[df2['Mean Price'] != 0]
    df2 = df2[df2['Volume'] != 0]
    df4 = df2[['Mean Price','Volume', '%_change', 'index2']]
    df3 = df2[['Mean Price','Volume', '%_change']]
    days_out = int(math.ceil(0.01 * len(df3)))
    df3['predictor'] = df3['Mean Price'].shift(-days_out)
    df3.dropna(inplace=True)
    x = df3.drop(['predictor'], 1)
    x = preprocessing.scale(x)
    x_nextMonth = x[-days_out:]
    df3.dropna(inplace=True)
    y = df3['predictor']


    ls = LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    ls.fit(x_train, y_train)
    score = ls.score(x_test, y_test)
    print "Accuracy of model is: ", score


    lasso = Lasso()
    svm_object = svm.SVR()
    print  "Lasso Accuracy:", cross_val_score(lasso, x, y, cv=10, scoring='mean_squared_error').mean()/-100
    print  "SVM Accuracy:",  cross_val_score(svm_object, x, y, cv=10, scoring='mean_squared_error').mean()/-100
    predict = ls.predict(x_nextMonth)
    print "Predicting mean price (average of open + close) of ", sys.argv[1], " for 1 month out "
    print predict


    title = sys.argv[1] + ' Price and Volume'
    df5 = df2[['Mean Price', 'Volume', 'index2']]
    df5.plot(subplots = True, figsize = (8, 8))
    plt.legend(loc = 'best')
    plt.title(title)
    plt.show()




def create_df(table_header, table_data):
    df = pd.DataFrame.from_dict(table_data, orient='index')
    df['index2'] = df.index
    # print df
    return df

def main():
    if len(sys.argv) >= 2:
        stock = sys.argv[1]
        get_data(stock)
        parse_data(bucket['content'])
        df = create_df(table_header, table_data)
        dff = data_cleansing(df)
        linear_model(dff)
    else:
        traceback.print_tb()

if __name__ == '__main__':
    main()
