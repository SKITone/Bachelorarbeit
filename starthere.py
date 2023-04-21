import datetime

#Get Data

from configs import binance_config, binance_spot
from binance.client import Client
#from binance.spot import Spot
import csv
from os.path import exists

#Get Indicator
import backtrader as bt
import strat_indicators

#Set Validdata
import validdata
import statistic

#Set Neural Network
import setneuralnet

#Test Neural Network
import testneuralnet

def getdata(filename, coin, candlesize, fromdate, todate):

    client = Client(binance_config.API_KEY, binance_config.API_SECRET)

    dir = "data/"
    if not(exists(dir+filename+'.csv')):

        csvfile = open(dir+filename+'.csv', 'w', newline='')
        candlestick_writer = csv.writer(csvfile, delimiter=',')           

        if candlesize == '1M': interval=Client.KLINE_INTERVAL_1MINUTE
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

    addindicator(filename, coin, candlesize, fromdate, todate)


def addindicator(filename, coin, candlesize, fromdate, todate):

    dir = "indicator_csv/"
    if not(exists(dir+filename+'.csv')):

        cerebro = bt.Cerebro()

        interval = 0
        if candlesize == '1M': interval= 1
        if candlesize == "5M": interval = 5
        if candlesize == "15M": interval = 15
        if candlesize == "30M": interval = 30
        if candlesize == "1H": interval = 60

        fdate = fromdate.strftime('%Y-%m-%d')
        fdate = datetime.datetime.strptime(fdate, '%Y-%m-%d')
        tdate = todate.strftime('%Y-%m-%d')
        tdate = datetime.datetime.strptime(tdate, '%Y-%m-%d') 

        dir = "data/"
        data = bt.feeds.GenericCSVData(dataname=dir+filename+".csv", dtformat=2, compression=interval, timeframe=bt.TimeFrame.Minutes, fromdate=fdate, todate=tdate)
        cerebro.adddata(data)

        #cerebro.addstrategy(strat_indicators.CHATGPT, scoin = coin, time = candlesize, filename = filename, test = False)
        cerebro.addstrategy(strat_indicators.newindicator, scoin = coin, time = candlesize, filename = filename, test = False)
        #cerebro.addstrategy(strat_indicators.relativeindicator, scoin = coin, time = candlesize, filename = filename, test = False)

        cerebro.run()


def localize_floats(row):
    return [
        str(el).replace('.', ',') if isinstance(el, float) else el 
        for el in row
    ]

def main():

    coins = ["BTC","ETH","XRP", "BNB", "DOGE"]
    #coins = ["BTC"]

    candlesizes = ["5M","15M","30M","1H"]
    #candlesizes = ["5M"]

    tensors=[12, 24, 48]
    tensors=[4]

    testranges = [12, 24, 48]
    testranges = [4]

    stds = [0.1 , 0.3 , 0.5]
    stds = [0.1]

    Epochs =   3
    wdh = 1

    #takeprofs = [0.01, 0.02, 0.03, 0.05]
    #takeprofs = [0.5]

    cooldowns = [6]

    newcsv =   True
    testwith22 = False
    predictions = []
    aresults = []
    Startpoints = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    Startpoints = [6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    #Startpoints = [5, 6, 7, 10, 11, 12, 13, 14, 15, 16]

    yesterday = datetime.date.today() - datetime.timedelta(days=0)
    yesterday = yesterday.strftime("%d %b, %Y")

    fromdate = datetime.datetime.strptime("1 Jan, 2021", "%d %b, %Y")
    todate = datetime.datetime.strptime("30 Jun, 2022", "%d %b, %Y")
    #todate = datetime.datetime.strptime(yesterday, "%d %b, %Y")
    testfromdate=datetime.datetime.strptime("1 Jul, 2022", "%d %b, %Y")
    testtodate = datetime.datetime.strptime("31 Dec, 2022", "%d %b, %Y")
    #testtodate =  datetime.datetime.strptime(yesterday, "%d %b, %Y")

    if testwith22:
        #testfromdate=datetime.datetime.strptime("6 Jan, 2023", "%d %b, %Y")
        #testfromdate=datetime.datetime.strptime("1 Feb, 2023", "%d %b, %Y")
        #testfromdate=datetime.datetime.strptime("26 Mar, 2022", "%d %b, %Y")
        testfromdate=datetime.datetime.strptime("1 Jul, 2022", "%d %b, %Y")
        #testfromdate=datetime.datetime.strptime("20 Dec, 2022", "%d %b, %Y")
        testtodate = datetime.datetime.strptime(yesterday, "%d %b, %Y")
        #testtodate = datetime.datetime.strptime("31 Dec, 2022", "%d %b, %Y")
        #testtodate=datetime.datetime.strptime("1 Jul, 2022", "%d %b, %Y")


    for coin in coins:
        #try:
            aresults.append(coin)
            for cooldown in cooldowns:
                for candlesize in candlesizes:

                    filename = coin+'_'+candlesize+'_'+fromdate.strftime("%b%Y")+'_'+todate.strftime("%b%Y")
                    #filename = "SNX_15M_Jan2022_Dec2022.csv"
                    testfilename = coin+'_'+candlesize+'_'+testfromdate.strftime("%b%Y")+'_'+testtodate.strftime("%b%Y")
                    #testfilename = testcoin+'_'+candlesize+'_'+testfromdate.strftime("%b%Y")+'_'+testtodate.strftime("%b%Y")+'.csv'
                    getdata(filename, coin, candlesize, fromdate, todate)
                    getdata(testfilename, coin, candlesize, testfromdate, testtodate)
                    #getdata(testcoin, candlesize, testfromdate, testtodate)
                    col = 0

                    resultfilename = "results/results"+filename+".txt"
                    f = open(resultfilename, 'w', newline='')
                    f.write('############################################################################\n')
                    wrow = "Coin: "+coin+"     Candlesize: "+candlesize
                    f.write(wrow+"\n")
                    f.write("\n")

                    if newcsv:
                        csvfilename = "results/"+filename+".csv"
                        csvf = open(csvfilename, 'w', newline='')
                        writer = csv.writer(csvf, delimiter=';')
                        csvdata = [[0 for i in range(len(testranges)*len(tensors)*wdh)] for j in range(len(stds)*4)]
                        print(len(csvdata))
                        print(len(csvdata[0]))

                    for testrange in testranges:
                        takeprofs= statistic.gettakeprof(filename, testrange, stds)
                        for tensor in tensors:
                            for i in range(wdh):
                                row = 0
                                for idx, takeprof in enumerate(takeprofs):
                                    print("Testzeitraum: "+str(testrange)+"     Tensor: "+str(tensor)+"     Takeprof: "+str(takeprof)+"     Standardabweichung:"+str(((idx+1)*0.1)))
                                    if not(testwith22):
                                        valid = validdata.takeprof(filename, takeprof, testrange)
                                        #valid = validdata.percentchangeaftertime(filename, takeprof, testrange)
                                    modeleval=""
                                    if not(testwith22):
                                        modeleval = setneuralnet.neuralnet(filename, valid, tensor, Epochs, takeprof, testrange, Startpoints)
                                    cooldown = testrange//2
                                    #cooldown = 0
                                    print("Coin: "+coin+" Cooldown: "+str(cooldown))
                                    results = testneuralnet.testStrategie(filename, testfilename, tensor, testrange, takeprof, cooldown, Startpoints, testfromdate, testtodate) #[result,  [signals, trades, postrade, negtrade, money]]
                                    #results = testneuralnet.percentStrategieOneMin(coin, filename, testfilename, tensor, testrange, takeprof, cooldown, Startpoints, testfromdate, testtodate) #[result,  [signals, trades, postrade, negtrade, money]]
                                    #results = testneuralnet.percentStrategie(filename, testfilename, tensor, testrange, takeprof, cooldown, Startpoints, testfromdate, testtodate) #[result,  [signals, trades, postrade, negtrade, money]]
                                    #results = testneuralnet.dcaStrategie(filename, testfilename, tensor, testrange, takeprof, cooldown, Startpoints, testfromdate, testtodate) #[result,  [signals, trades, postrade, negtrade, money]]
                                    aresults.append(results[1][4])
                                    predictions.append(results[2])

                                    wrow = "Testzeitraum: "+str(testrange)+"     Tensor: "+str(tensor)+"     Modelevaluation: "+str(modeleval)+"     Takeprof/ Stoploss: "+str(takeprof)
                                    f.write(wrow+"\n")
                                    wrow = results[0]
                                    f.write(wrow+"\n")
                                    f.write("\n")
                                    
                                    f.write("\n")
                                    print(col)
                                    print(row)
                                    if newcsv:
                                        csvdata[row][col]=round(results[1][4]-100,2)      #percent
                                        csvdata[row+1][col]=int(results[1][1])     #trades
                                        csvdata[row+2][col]=int(results[1][2])     #positiv
                                    row += 4
                                col += 1
                    f.close()
                    if newcsv:
                        for row in csvdata:
                            writer.writerow(localize_floats(row))
                        csvf.close()
                
        # except Exception as e:
        #     print(coin+" tut nicht")
        #     print(e)
            print(aresults)

    

    
    #results = testneuralnet.multtestStrategie("test5", coins, testranges[0], predictions, tensors[0])
    print(aresults)

if __name__ == '__main__':
    main()
