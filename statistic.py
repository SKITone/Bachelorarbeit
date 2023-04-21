import datetime
from binance.client import Client
from configs import binance_config
from os.path import exists
import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

def getdata():

    coin = "ETH"
    candlesizes = ["5M", "15M", "30M", "1H"]
    fromdate = datetime.datetime.strptime( "1 Jan, 2021", "%d %b, %Y" )
    todate = datetime.datetime.strptime( "31 Dec, 2021", "%d %b, %Y" )

    client = Client(binance_config.API_KEY, binance_config.API_SECRET)

    start = fromdate.strftime("%b%Y")
    end = todate.strftime("%b%Y")

    print('coin '+str(coin))

    for candlesize in candlesizes:

        filename = coin+'_'+candlesize+'_'+start+'_'+end+'.csv'                                         #filename

        if not(exists('data/'+coin+'_'+candlesize+'_'+start+'_'+end+'.csv')):

            csvfile = open('data/'+filename, 'w', newline='')
            candlestick_writer = csv.writer(csvfile, delimiter=',')
                
            if candlesize == '5M': interval=Client.KLINE_INTERVAL_5MINUTE
            if candlesize == '15M': interval=Client.KLINE_INTERVAL_15MINUTE
            if candlesize == '30M': interval=Client.KLINE_INTERVAL_30MINUTE
            if candlesize == '1H': interval=Client.KLINE_INTERVAL_1HOUR

            print("downloading "+str(coin)+' '+str(candlesize))
            candlesticks = client.get_historical_klines(coin+"USDT", interval, str(fromdate), str(todate))
            print("writing csv file")
            for candlestick in candlesticks:
                candlestick[0] = candlestick[0] / 1000 
                candlestick_writer.writerow(candlestick)
            csvfile.close()
        else:
            print("file exists")

def distribution():
    
    coin = "ETH"
    candlesizes = ["5M",  "15M", "30M", "1H"]
    fromdate = datetime.datetime.strptime("1 Jan, 2021", "%d %b, %Y")
    todate = datetime.datetime.strptime("31 Dec, 2021", "%d %b, %Y")
    begin = fromdate.strftime("%b%Y")
    end = todate.strftime("%b%Y")

    for candlesize in candlesizes:

        filename = "data/"+coin+'_'+candlesize+'_'+begin+'_'+end+'.csv'

        df = pd.read_csv(filename, header=0)

        dis = []

        df = df.to_numpy()

        priceclose = 4

        testranges = [4, 8, 16, 32, 64, 128, 256, 512]

        standarddevs = [[0 for i in range(8)] for j in range(12)]

        count = 0

        cut = 0

        for testrange in testranges:
            
            for i in range(1, len(df)-(testrange+1)):

                max = df[i,priceclose]
                min = df[i,priceclose]
                start = df[i,priceclose]

                for j in range(i+1, i+(testrange+1)):
                    if max < df[j,priceclose]:
                        max = df[j,priceclose]
                    if min > df[j,priceclose]:
                        min = df[j,priceclose]
                max = ((max-start)*100)/start
                min = ((min-start)*100)/start
                if abs(max) > abs(min):
                    start = max
                else:
                    start = min
                dis.append(start)
            #pd.DataFrame(dis).to_csv('MinMax.csv')

            dis.sort()
            mean=np.mean(dis)
            sd = np.std(dis)        #Standardabweichung

            standarddevs[0][count]=round(sd*0.1,2)
            standarddevs[1][count]=round(sd*0.2,2)
            standarddevs[2][count]=round(sd*0.3,2)
            standarddevs[3][count]=round(sd*0.4,2)
            standarddevs[4][count]=round(sd*0.5,2)
            standarddevs[5][count]=round(sd*0.6,2)
            standarddevs[6][count]=round(sd*0.8,2)
            standarddevs[7][count]=round(sd*1,2)
            standarddevs[8][count]=round(sd*1.5,2)
            standarddevs[9][count]=round(sd*2,2)
            standarddevs[10][count]=round(sd*2.5,2)
            standarddevs[11][count]=round(sd*3,2)
            count +=1

            cut = sd * 3

            # prob = norm.cdf(3*sd, loc = mean, scale = sd) - norm.cdf(-3*sd, loc = mean, scale = sd)   #loc = location = mean, scale = standard deviation
            # print(type(prob))
            # print(prob)

            # print("0,5 Standardabweichung: "+ str(round(sd/2,2)) + " %")
            # print("1 Standardabweichung: "+ str(round(sd,2)) + " %")
            # print("1,5 Standardabweichung: "+ str(round(sd*1.5,2)) + " %")
            # print("2 Standardabweichung: "+ str(round(sd*2,2)))
            # print("2,5 Standardabweichung: "+ str(round(sd*2.5,2)) + " %")
            # print("3 Standardabweichung: "+ str(round(sd*3,2)) + " %")


            pdf = norm.pdf(dis, mean, sd)
        
        #     plt.plot(dis, pdf , label= str(testrange) )
        # plt.title(coin + " " + candlesize + " Standardabweichung", fontsize=14)
        # plt.legend(title = 'Testzeiträume')
        # plt.ylabel('Dichte')
        # plt.xlabel('Prozent')
        # plt.xlim(-cut, cut)
        # plt.savefig('stddevs/standarddev_'+coin+'_'+candlesize+'.png')
        # plt.show()
        pd.DataFrame(standarddevs).to_csv('stddevs/standarddev_'+coin+'_'+candlesize+'.csv', header = False, index = False)



def gettakeprof(filename, testrange, stds):

    dir = "data/"
    df = pd.read_csv(dir+filename+'.csv', header=0)
    dis = []
    df = df.to_numpy()
    priceclose = 4


    standarddevs = []

    for i in range(1, len(df)-(testrange+1)):

        max = df[i,priceclose]
        min = df[i,priceclose]
        start = df[i,priceclose]

        for j in range(i+1, i+(testrange+1)):
            if max < df[j,priceclose]:
                max = df[j,priceclose]
            if min > df[j,priceclose]:
                min = df[j,priceclose]
        max = ((max-start)*100)/start
        min = ((min-start)*100)/start
        if abs(max) > abs(min):
            start = max
        else:
            start = min
        dis.append(start)
    sd = np.std(dis)        #Standardabweichung

    for std in stds:
        standarddevs.append(round(sd*std,2))

    return standarddevs


if __name__ == "__main__":
    getdata()
    distribution()