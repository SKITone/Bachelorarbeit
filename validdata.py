import numpy as np
import pandas as pd
import csv

priceopen = 1
priceclose = 4
pricelow = 3
pricehigh = 2
atr = 5
atrscale = 3    #average true range (volatility)
percent = 5     #how much win

def percentchangeaftertime(filename, takeprof, validrange):
    
    dir = "indicator_csv/"
    df = pd.read_csv(dir+filename+'.csv', header=0)

    Y = []

    df = df.to_numpy()

    percent = takeprof

    for i in range(1, len(df)-validrange):
        buyprice = df[i,priceclose]
        takeprof = buyprice*((100+percent)/100)
        stoploss = buyprice*((100-percent)/100)
        for j in range(i+1, i+validrange):
            if takeprof < df[j,priceclose]:
                Y.append([1,0,0])
                break
            if stoploss > df[j,priceclose]:
                Y.append([0,0,1])
                break
            if j == i+validrange-1:
                Y.append([0,1,0])
    print(len(Y))
    return Y

def candlecolor(filename):
    
    df = pd.read_csv(filename, header=0) #load candles with indicators

    Y = []

    df = df.to_numpy()

    for i in range(1, len(df)-1):
        diff = (df[i,priceclose] - df[i,priceopen]) / df[i,priceopen]
        Y.append(diff)
    return Y

def takeprof(filename, takeprof, validrange):
    
    dir = "indicator_csv/"
    df = pd.read_csv(dir+filename+'.csv', header=0)

    Y = []

    df = df.to_numpy()

    for i in range(0, len(df)-(validrange)):
        #if i % 10000 == 0:
            #print(i)
        #buyprice = df[i,priceclose]
        max = df[i,priceclose] * (1+(takeprof/100))
        min = df[i,priceclose] * (1-(takeprof/100))
        count = 0
        for j in range(i+1, i+validrange):
            count += 1
            if max < df[j,priceclose]:
                Y.append([1,0,0])
                break
            if min > df[j,priceclose]:
                Y.append([0,0,1])
                break
            if j == i+validrange-1:
                Y.append([0,1,0])
    return Y