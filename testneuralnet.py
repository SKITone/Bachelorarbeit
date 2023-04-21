import backtrader as bt
import btstrat
from keras import models
import numpy as np 
import pandas as pd
import csv
import numpy as np
import datetime
import starthere


def testStrategie(filename, testfilename, LSTMTensor, testrange, takeprof, cooldown, Startpoints, testfromdate, testtodate):

    dir = "indicator_csv/"
    testdata = dir+testfilename+".csv"


    new_model = models.load_model('neural_networks/'+filename)
    #new_model = models.load_model('neural_networks/DOGE_5M_Jan2021_Jun2022')
    #new_model = models.load_model('live_neuralnets/DOGE_5M_Jan2021_Jun2022')
    #new_model = models.load_model('live_neuralnets/new')

    df = pd.read_csv(testdata)
    df = df.to_numpy()
    df = df[:, Startpoints]
    Startpoints = df.shape[1]

    X = []
    priceclose = 4

    for i in range(0, len(df)):
        X.append(np.array(df[i]))

    X = np.array(X).reshape(-1, Startpoints)

    X = np.asarray(X).astype(np.float32)

    X3D = []

    for i in range(LSTMTensor, len(X)):
        X3D.append(np.array(X[i-LSTMTensor:i]))


    X3D = np.asarray(X3D).reshape(-1, LSTMTensor, Startpoints)

    predictions = new_model.predict(X3D, verbose = 0)

    df = pd.read_csv(testdata)

    df = np.array(df)

    df = np.delete( df, range(0, LSTMTensor) , 0)

    money = 100

    pos = False

    signals = 0
    trades=0
    postrade=0
    negtrade=0

    # #only one pos
    mean = np.mean(predictions, axis=0)
    print(mean[0])
    print(mean[1])
    print(mean[2])
    count = 0
    fee = 0.000

    #f = open(filename, 'w', newline='')
    buytimes = []
    selltimes = []

    for i in range(0,len(predictions)):
        count +=1

        if predictions[i,0] > predictions[i,1] and predictions[i,0] > predictions[i,2]:
            signals +=1
            if not(pos):
                buyprice = df[i,priceclose]*(1 + fee) 
                trades +=1
                pos = True
                time = df[i,0]
                count = 0
                buytimes.append(time)

        if pos and (count > cooldown or i == len(predictions)-1):
            if ((predictions[i,2] > predictions[i,1] and predictions[i,2] > predictions[i,0])) or (i == len(predictions)-1):
                money += (money/100) * ((df[i,priceclose]*(1 - fee)  - buyprice) / buyprice)*100
                print('Open Time: '+ time+  ' Sell Time: ',df[i,0]+ ' Change '+ '% 0.2f' % float(((float(df[i,priceclose])*(1 - fee) - buyprice) / buyprice )*100) + ' sell:'+ '% 0.5f' % float(df[i,priceclose]*(1 - fee)),' buy:'+ '% 0.5f' % buyprice+ " percent:"+ '% 0.2f' %  money)
                #f.write('Open Time: '+ time+  ' Sell Time: '+df[i,0]+ ' Change '+ '% 0.2f' % float(((float(df[i,priceclose])*(1 - fee) - buyprice) / buyprice )*100) + ' sell:'+ '% 0.5f' % float(df[i,priceclose]*(1 - fee))+' buy:'+ '% 0.5f' % buyprice+ " percent:"+ '% 0.2f' %  money+"\n")
                pos = False
                time = df[i,0]
                time = df[i,0]
                selltimes.append(time)
                if buyprice > df[i,priceclose]:
                    negtrade +=1
                else:
                    postrade +=1
    #f.close

    #BacktraderTest(testfilename, buytimes, selltimes, testfromdate, testtodate)
        
    print()
    print("Signale:",str(signals)," Trades:", trades, " postive Trades:", postrade, " negative Trades:", negtrade, " Prozent:", money)
    print()
    print('##########################################################################')
    print()
    print()

    result =  "Signale: "+str(signals)+"     Trades: "+ str(trades)+ "     postive Trades: "+ str(postrade)+ "     negative Trades: "+ str(negtrade)+ "     Prozent: "+ str(money)   
    return [result,  [signals, trades, postrade, negtrade, money], predictions]



def BacktraderTest(filename, buytimes, selltimes, testfromdate, testtodate):
    cerebro = bt.Cerebro()

    #testfromdate = datetime.datetime.strptime(testfromdate, '%Y-%m-%d')
    #testtodate = datetime.datetime.strptime(testtodate, '%Y-%m-%d')

    dir = "data/"

    data = bt.feeds.GenericCSVData(dataname=dir+filename+".csv", dtformat=2, compression=5, timeframe=bt.TimeFrame.Minutes, fromdate=testfromdate, todate=testtodate)

    cerebro.adddata(data)

    cerebro.addstrategy(btstrat.timebuy, buytimes=buytimes, selltimes =selltimes)

    cerebro.broker.setcommission(commission=0.000)

    cerebro.addsizer(bt.sizers.PercentSizer, percents = 99) #default 20%

    cerebro.broker.set_cash(100)

    print('Starting Portfolio Value: %.2f ' % cerebro.broker.getvalue())

    cerebro.run()

    print('Final Portfolio Value: %.2f ' % cerebro.broker.getvalue())

    cerebro.plot()




def reltestStrategie(filename, testfilename, LSTMTensor, testrange, takeprof, cooldown, Startpoints):

    dir = "indicator_csv/"
    testdata = dir+testfilename+".csv"

    #new_model = models.load_model('neural_networks/'+filename)
    #new_model = models.load_model('live_neuralnets/TWT_5M_Jan2021_Jun2022')
    new_model = models.load_model('live_neuralnets/DOGE_1H_Jan2021_Jun2022')

    df = pd.read_csv(testdata)
    df = df.to_numpy()

    df = df[:, Startpoints]
    Startpoints = df.shape[1]

    X = []
    priceclose = 4

    for i in range(0, len(df)):
        X.append(np.array(df[i]))

    X = np.array(X).reshape(-1, Startpoints)

    X = np.asarray(X).astype(np.float32)

    X3D = []

    for i in range(LSTMTensor, len(X)):
        X3D.append(np.array(X[i-LSTMTensor:i]))


    X3D = np.asarray(X3D).reshape(-1, LSTMTensor, Startpoints)

    predictions = new_model.predict(X3D, verbose = 0)

    df = pd.read_csv(testdata)

    df = np.array(df)

    df = np.delete( df, range(0, LSTMTensor) , 0)


    # cerebro = bt.Cerebro()

    # data = bt.feeds.GenericCSVData(dataname="data/"+filename, dtformat=2, compression=int(candlesize[:-1]), timeframe=bt.TimeFrame.Minutes, fromdate=testfromdate, todate=testtodate)

    # cerebro.adddata(data)


    # cerebro.addstrategy(btstrat.timebuy, buytimes=times)

    # cerebro.addsizer(bt.sizers.PercentSizer, percents = 99) #default 20%

    # cerebro.broker.set_cash(1000)

    # print('Starting Portfolio Value: %.2f ' % cerebro.broker.getvalue())

    # cerebro.run()

    # print('Final Portfolio Value: %.2f ' % cerebro.broker.getvalue())

    # #cerebro.plot()

    # #only one pos

    csvpred = pd.DataFrame(predictions)
    csvpred.to_csv('pred.csv')


    stds = [0.7, 0.8, 0.9, 0.95, 1, 1.1, 1.2, 1.4, 1.6, 1.8, 2]







    count = 0
    fee = 0.000

    #f = open(filename, 'w', newline='')
    for std in stds:
        predkauf = []
        predverkauf = []

        for prediction in predictions:
            predkauf.append(prediction[0])
            predverkauf.append(prediction[2])
        kaufmean = np.mean(predkauf)
        verkaufmean = np.mean(predverkauf)
        print(kaufmean)
        print(verkaufmean)
        sdkauf = np.std(predkauf)
        sdverkauf = np.std(predverkauf)
        print(sdkauf)
        print(sdverkauf)
        kauf = round(kaufmean+sdkauf*std,8)
        print(kauf)
        verkauf = round(verkaufmean+sdverkauf*std,8)
        print(verkauf)

        money = 100

        pos = False

        signals = 0
        trades=0
        postrade=0
        negtrade=0
        for i in range(0,len(predictions)):
            count -=1
            #print(predictions[i,0])
            #print(kauf)
            if predictions[i,0] > kauf:
                signals +=1
                if not(pos):
                    buyprice = df[i,priceclose]*(1 + fee) 
                    trades +=1
                    pos = True
                    time = df[i,0]
                    count = cooldown

            if pos and (count < 0 or i == len(predictions)-1):
                if (predictions[i,2] > verkauf) or (i == len(predictions)-1):
                    money += (money/100) * ((df[i,priceclose]*(1 - fee)  - buyprice) / buyprice)*100
                    print('Open Time: '+ time+  ' Sell Time: ',df[i,0]+ ' Change '+ '% 0.2f' % float(((float(df[i,priceclose])*(1 - fee) - buyprice) / buyprice )*100) + ' sell:'+ '% 0.5f' % float(df[i,priceclose]*(1 - fee)),' buy:'+ '% 0.5f' % buyprice+ " percent:"+ '% 0.2f' %  money)
                    #f.write('Open Time: '+ time+  ' Sell Time: '+df[i,0]+ ' Change '+ '% 0.2f' % float(((float(df[i,priceclose])*(1 - fee) - buyprice) / buyprice )*100) + ' sell:'+ '% 0.5f' % float(df[i,priceclose]*(1 - fee))+' buy:'+ '% 0.5f' % buyprice+ " percent:"+ '% 0.2f' %  money+"\n")
                    pos = False
                    if buyprice > df[i,priceclose]:
                        negtrade +=1
                    else:
                        postrade +=1
        #f.close


            
        print()
        print("Signale:",str(signals)," Trades:", trades, " postive Trades:", postrade, " negative Trades:", negtrade, " Prozent:", money)
        print()
        print('##########################################################################')
        print()
        print()

        result =  "Signale: "+str(signals)+"     Trades: "+ str(trades)+ "     postive Trades: "+ str(postrade)+ "     negative Trades: "+ str(negtrade)+ "     Prozent: "+ str(money)   
    return [result,  [signals, trades, postrade, negtrade, money], predictions]

def dcaStrategie(filename, testfilename, LSTMTensor, testrange, takeprof, cooldown, Startpoints, testfromdate, testtodate):

    dir = "indicator_csv/"
    testdata = dir+testfilename+".csv"


    #new_model = models.load_model('neural_networks/'+filename)
    #new_model = models.load_model('neural_networks/DOGE_5M_Jan2021_Jun2022')
    new_model = models.load_model('live_neuralnets/DOGE_5M_Jan2021_Jun2022')
    #new_model = models.load_model('live_neuralnets/new')

    df = pd.read_csv(testdata)
    df = df.to_numpy()
    df = df[:, Startpoints]
    Startpoints = df.shape[1]

    X = []
    priceclose = 4

    for i in range(0, len(df)):
        X.append(np.array(df[i]))

    X = np.array(X).reshape(-1, Startpoints)

    X = np.asarray(X).astype(np.float32)

    X3D = []

    for i in range(LSTMTensor, len(X)):
        X3D.append(np.array(X[i-LSTMTensor:i]))


    X3D = np.asarray(X3D).reshape(-1, LSTMTensor, Startpoints)

    predictions = new_model.predict(X3D, verbose = 0)

    df = pd.read_csv(testdata)

    df = np.array(df)

    df = np.delete( df, range(0, LSTMTensor) , 0)

    money = 100

    predkauf = []
    predverkauf = []
    diff = []

    for prediction in predictions:
        diff.append(prediction[0]-prediction[2])
    mean = np.mean(diff)
    sd = np.std(diff)
    kauf = round((mean+sd*1)*100,8)
    print(str(kauf))

    pos = -1
    buyprice = 0
    signals = 0
    trades=0
    postrade=0
    negtrade=0
    count = 0
    priceclose = 4
    loopagain=True
    fee= 0.001
    testcount = 0
    takeprofit = 0
    stoploss = 0
    buytimes = []
    selltimes = []
    takestopperc = 0.02
    dca = 0
    buyprice0 = 0
    buyprice1 = 0
    buyprice2 = 0

    f = open("results/"+filename, 'w', newline='')

    for i, prediction in enumerate(predictions):
        #if prediction[i,0] > prediction[i,1] and prediction[i,0] > prediction[i,2]:
        #print("Pred: "+str(round((prediction[i,0]-prediction[i,2])*100,2)))
        if round((prediction[0]-prediction[2])*100,2) > 11: #default 11
            signals +=1
            if pos == -1:
                buyprice = df[i,priceclose]
                buyprice0 = df[i,priceclose]
                takeprofit = buyprice * (1 + takestopperc)
                stoploss = buyprice * (1 - takestopperc)
                trades +=1
                pos = 0
                dca = 0
                time = df[i,0]
                buytimes.append(time)

        #if pos == index and count > testrange*4:-4
        #if df[i,priceclose] > buyprice and round((prediction[0]-prediction[2])*100,2) > 10:
        if pos == 0 and round((prediction[0]-prediction[2])*100,2) > 10: #default 10
                takeprofit = df[i,priceclose] * 1.02
                stoploss = df[i,priceclose] * 0.98
                if df[i,priceclose] < buyprice0 and dca == 0:
                    buyprice1 = df[i,priceclose]
                    dca += 1
                if df[i,priceclose] < buyprice1 and dca == 1:
                    buyprice2 = df[i,priceclose]
                    dca += 1

        #if pos == 0 and (df[i,priceclose] > takeprofit or i == len(predictions)-1):
        #if pos == 0 and ( ( round((prediction[0]-prediction[2])*100,2) < -4 and df[i,priceclose] > takeprofit ) or (round((prediction[0]-prediction[2])*100,2) < 0 and df[i,priceclose] < stoploss ) ) or i == len(predictions)-1:
        if pos == 0 and ( round((prediction[0]-prediction[2])*100,2) < -4.1 and df[i,priceclose] > takeprofit) or i == len(predictions)-1: #default -4.1
            testcount += 1
            #if ((prediction[i,2] > prediction[i,1] and prediction[i,2] > prediction[i,0])):
            #if round((prediction[0]-prediction[2])*100,2) > -7 or df[i,priceclose] < stoploss: 
            if True:
                if dca == 0:
                    profit = (df[i,priceclose]*(1 - fee*2) / buyprice0) - 1
                    money = money * (1 + (profit*0.25))
                if dca == 1:
                    profit = (buyprice1*(1 - fee*2) / buyprice0)  - 1
                    money = money * (1 + (profit*0.25))
                    profit = (df[i,priceclose]*(1 - fee*2) / buyprice1)  - 1
                    money = money * (1 + (profit*0.5))
                if dca == 2:
                    profit = (buyprice1*(1 - fee*2) / buyprice0)  - 1
                    money = money * (1 + (profit*0.25))
                    profit = (buyprice2*(1 - fee*2) / buyprice1)  - 1
                    money = money * (1 + (profit*0.5))
                    profit = (df[i,priceclose]*(1 - fee*2) / buyprice2)  - 1
                    money = money * (1 + profit)


                print('Open Time:'+ time+  ' Sell Time:',df[i,0]+ ' Change'+ '% 0.2f' % float(((float(df[i,priceclose])*(1 - fee) - buyprice) / buyprice )*100) + ' sell:'+ '% 0.5f' % float(df[i,priceclose]*(1 - fee)),' buy:'+ '% 0.5f' % buyprice+ " percent:"+ '% 0.2f' %  money)
                #f.write('Open Time:'+ time+  ' Sell Time:'+df[i,0]+ ' Change'+ '% 0.2f' % float(((float(df[i,priceclose])*(1 - fee) - buyprice) / buyprice )*100) + ' sell:'+ '% 0.5f' % float(df[i,priceclose]*(1 - fee))+' buy:'+ '% 0.5f' % buyprice+ " percent:"+ '% 0.2f' %  money+"\n")
                pos = -1
                time = df[i,0]
                selltimes.append(time)
                #loopagain = True
                stoploss = 0
                armed = False
                if buyprice > df[i,priceclose]:
                    negtrade +=1
                else:
                    postrade +=1
    f.close

    
    print()
    print("Coin Change Over Time: "+ str((1-(df[0,priceclose]/df[len(df)-1,priceclose]))*100))
    print()
    print("Signale:",str(signals)," Trades:", trades, " postive Trades:", postrade, " negative Trades:", negtrade, " Prozent:", money)
    print()
    print('##########################################################################')
    print()
    print()

    BacktraderTest(testfilename, buytimes, selltimes, testfromdate, testtodate)

    result =  "Signale: "+str(signals)+"     Trades: "+ str(trades)+ "     postive Trades: "+ str(postrade)+ "     negative Trades: "+ str(negtrade)+ "     Prozent: "+ str(money)   
    return [result,  [signals, trades, postrade, negtrade, money], predictions]

def percentStrategie(filename, testfilename, LSTMTensor, testrange, takeprof, cooldown, Startpoints, testfromdate, testtodate):

    dir = "indicator_csv/"
    testdata = dir+testfilename+".csv"


    #new_model = models.load_model('neural_networks/'+filename)
    #new_model = models.load_model('neural_networks/DOGE_5M_Jan2021_Jun2022')
    new_model = models.load_model('live_neuralnets/DOGE_5M_Jan2021_Jun2022')
    #new_model = models.load_model('live_neuralnets/new')

    df = pd.read_csv(testdata)
    df = df.to_numpy()
    df = df[:, Startpoints]
    Startpoints = df.shape[1]

    X = []
    priceclose = 4

    for i in range(0, len(df)):
        X.append(np.array(df[i]))

    X = np.array(X).reshape(-1, Startpoints)

    X = np.asarray(X).astype(np.float32)

    X3D = []

    for i in range(LSTMTensor, len(X)):
        X3D.append(np.array(X[i-LSTMTensor:i]))


    X3D = np.asarray(X3D).reshape(-1, LSTMTensor, Startpoints)

    predictions = new_model.predict(X3D, verbose = 0)

    df = pd.read_csv(testdata)

    df = np.array(df)

    df = np.delete( df, range(0, LSTMTensor) , 0)

    money = 100

    predkauf = []
    predverkauf = []
    diff = []

    for prediction in predictions:
        diff.append(prediction[0]-prediction[2])
    mean = np.mean(diff)
    sd = np.std(diff)
    kauf = round((mean+sd*1)*100,8)
    print(str(kauf))

    pos = -1
    buyprice = 0
    signals = 0
    trades=0
    postrade=0
    negtrade=0
    count = 0
    priceclose = 4
    loopagain=True
    fee= 0.001
    testcount = 0
    takeprofit = 0
    stoploss = 0
    buytimes = []
    selltimes = []
    takestopperc = 0.02

    f = open("results/"+filename, 'w', newline='')

    for i, prediction in enumerate(predictions):
        #if prediction[i,0] > prediction[i,1] and prediction[i,0] > prediction[i,2]:
        #print("Pred: "+str(round((prediction[i,0]-prediction[i,2])*100,2)))
        if round((prediction[0]-prediction[2])*100,2) > 11: #default 11
            signals +=1
            if pos == -1:
                buyprice = df[i,priceclose]
                takeprofit = buyprice * (1 + takestopperc)
                stoploss = buyprice * (1 - takestopperc)
                trades +=1
                pos = 0
                time = df[i,0]
                buytimes.append(time)

        #if pos == index and count > testrange*4:-4
        #if df[i,priceclose] > buyprice and round((prediction[0]-prediction[2])*100,2) > 10:
        #if pos == 0 and round((prediction[0]-prediction[2])*100,2) > 10: #default 10
        if False:
                takeprofit = df[i,priceclose] * 1.02
                stoploss = df[i,priceclose] * 0.98
        #if pos == 0 and (df[i,priceclose] > takeprofit or i == len(predictions)-1):
        #if pos == 0 and ( ( round((prediction[0]-prediction[2])*100,2) < -4 and df[i,priceclose] > takeprofit ) or (round((prediction[0]-prediction[2])*100,2) < 0 and df[i,priceclose] < stoploss ) ) or i == len(predictions)-1:
        #if pos == 0 and ( round((prediction[0]-prediction[2])*100,2) < -4.1 and ((df[i,priceclose] > takeprofit) or df[i,priceclose] < stoploss )) or i == len(predictions)-1:
        if pos == 0 and ( round((prediction[0]-prediction[2])*100,2) < -4.1  or i == len(predictions)-1):
        #if pos == 0 and ( round((prediction[0]-prediction[2])*100,2) < -4.1 and df[i,priceclose] > takeprofit) or i == len(predictions)-1: #default -4.1
        #if pos == 0 and  round((prediction[0]-prediction[2])*100,2) < -4.1 :    
            testcount += 1
            #if ((prediction[i,2] > prediction[i,1] and prediction[i,2] > prediction[i,0])):
            #if round((prediction[0]-prediction[2])*100,2) > -7 or df[i,priceclose] < stoploss: 
            if True:
                money += ((money/100) * ((df[i,priceclose]*(1 - fee*2) - buyprice) / buyprice)*100)
                print('Open Time:'+ time+  ' Sell Time:',df[i,0]+ ' Change'+ '% 0.2f' % float(((float(df[i,priceclose])*(1 - fee) - buyprice) / buyprice )*100) + ' sell:'+ '% 0.5f' % float(df[i,priceclose]*(1 - fee)),' buy:'+ '% 0.5f' % buyprice+ " percent:"+ '% 0.2f' %  money)
                #f.write('Open Time:'+ time+  ' Sell Time:'+df[i,0]+ ' Change'+ '% 0.2f' % float(((float(df[i,priceclose])*(1 - fee) - buyprice) / buyprice )*100) + ' sell:'+ '% 0.5f' % float(df[i,priceclose]*(1 - fee))+' buy:'+ '% 0.5f' % buyprice+ " percent:"+ '% 0.2f' %  money+"\n")
                pos = -1
                time = df[i,0]
                selltimes.append(time)
                #loopagain = True
                stoploss = 0
                armed = False
                if buyprice > df[i,priceclose]:
                    negtrade +=1
                else:
                    postrade +=1
    f.close

    
    print()
    print("Coin Change Over Time: "+ str((1-(df[0,priceclose]/df[len(df)-1,priceclose]))*100))
    print()
    print("Signale:",str(signals)," Trades:", trades, " postive Trades:", postrade, " negative Trades:", negtrade, " Prozent:", money)
    print()
    print('##########################################################################')
    print()
    print()

    BacktraderTest(testfilename, buytimes, selltimes, testfromdate, testtodate)

    result =  "Signale: "+str(signals)+"     Trades: "+ str(trades)+ "     postive Trades: "+ str(postrade)+ "     negative Trades: "+ str(negtrade)+ "     Prozent: "+ str(money)   
    return [result,  [signals, trades, postrade, negtrade, money], predictions]


def percentStrategieOneMin(coin, filename, testfilename, LSTMTensor, testrange, takeprof, cooldown, Startpoints, testfromdate, testtodate):

    filenameOneMin = coin+'_1M_'+testfromdate.strftime("%b%Y")+'_'+testtodate.strftime("%b%Y")
    
    starthere.getdata(filenameOneMin, coin, "1M", testfromdate, testtodate)

    dir = "indicator_csv/"
    testdata = dir+testfilename+".csv"

    testdataOneMin = dir+filenameOneMin+".csv"


    #new_model = models.load_model('neural_networks/'+filename)
    #new_model = models.load_model('neural_networks/DOGE_5M_Jan2021_Jun2022')
    new_model = models.load_model('live_neuralnets/DOGE_5M_Jan2021_Jun2022')
    #new_model = models.load_model('live_neuralnets/new')

    df = pd.read_csv(testdata, header = 0)
    df = df.to_numpy()
    firstdate = df[0][0]
    print(firstdate)
    df = df[:, Startpoints]
    dfOneMin = pd.read_csv(testdataOneMin, header = None)
    dfOneMin = dfOneMin.to_numpy()

    for i in range(0, len(dfOneMin)):
        if firstdate == dfOneMin[i][0]:
            deletecols = i
            break

    dfOneMin = dfOneMin[deletecols:len(dfOneMin)]

    print(dfOneMin[0][0])
    dfOneMin = dfOneMin[:, Startpoints]
    Startpoints = df.shape[1]

    X = []
    XOneMin = []
    priceclose = 4

    for i in range(0, len(df)):
        X.append(np.array(df[i]))
    for i in range(0, len(dfOneMin)):
        XOneMin.append(np.array(dfOneMin[i]))

    X = np.array(X).reshape(-1, Startpoints)
    XOneMin = np.array(XOneMin).reshape(-1, Startpoints)

    X = np.asarray(X).astype(np.float32)
    XOneMin = np.asarray(XOneMin).astype(np.float32)

    X3D = []

    for i in range(LSTMTensor*5, len(XOneMin)):
        List = X[i//5-LSTMTensor:i//5-1]
        List = np.append(List,np.array([XOneMin[i]]))
        try:
            List = np.array(List).reshape(LSTMTensor, Startpoints)
            X3D.append(List)
        except Exception as e:
            break


    X3D = np.asarray(X3D).reshape(-1, LSTMTensor, Startpoints)
    print(str(X3D[0][3]))

    predictions = new_model.predict(X3D, verbose = 0)

    dfOneMin = pd.read_csv(testdataOneMin, header = None)

    dfOneMin = np.array(dfOneMin)

    dfOneMin = np.delete( dfOneMin, range(0, deletecols + LSTMTensor*5) , 0)

    print(str(len(predictions)))
    print(str(len(dfOneMin)))
    print(dfOneMin[0])

    money = 100

    predkauf = []
    predverkauf = []
    diff = []

    for prediction in predictions:
        diff.append(prediction[0]-prediction[2])
    mean = np.mean(diff)
    sd = np.std(diff)
    kauf = round((mean+sd*1)*100,8)
    print(str(kauf))

    pos = -1
    buyprice = 0
    signals = 0
    trades=0
    postrade=0
    negtrade=0
    count = 0
    priceclose = 4
    loopagain=True
    fee= 0.001
    testcount = 0
    takeprofit = 0
    stoploss = 0
    buytimes = []
    selltimes = []

    f = open("results/"+filename, 'w', newline='')

    for i, prediction in enumerate(predictions, 20):
        #if prediction[i,0] > prediction[i,1] and prediction[i,0] > prediction[i,2]:
        #print("Pred: "+str(round((prediction[i,0]-prediction[i,2])*100,2)))
        if round((prediction[0]-prediction[2])*100,2) > 11:
            signals +=1
            if pos == -1:
                buyprice = dfOneMin[i,priceclose]
                buypricenew = dfOneMin[i,priceclose]
                takeprofit = buyprice * 1.02
                stoploss = buyprice * 0.98
                trades +=1
                pos = 0
                time = dfOneMin[i,0]
                buytimes.append(time)

        #if pos == index and count > testrange*4:-4
        #if round((prediction[0]-prediction[2])*100,2) > 11 and dfOneMin[i,priceclose] < buypricenew:
        #if pos == 0 and round((prediction[0]-prediction[2])*100,2) > 10:
        if False:
                takeprofit = dfOneMin[i,priceclose] * 1.02
                stoploss = dfOneMin[i,priceclose] * 0.98
                buypricenew = dfOneMin[i,priceclose]
        #if pos == 0 and (dfOneMin[i,priceclose] > takeprofit or i == len(predictions)-1):

        if pos == 0 and ( round((prediction[0]-prediction[2])*100,2) < -4.1 and ((dfOneMin[i,priceclose] > takeprofit) or dfOneMin[i,priceclose] < stoploss )) or i == len(predictions)-1:
        #if pos == 0 and ( round((prediction[0]-prediction[2])*100,2) < -4.1  or i == len(predictions)-1):
        #if pos == 0 and ( round((prediction[0]-prediction[2])*100,2) < -4.1 and dfOneMin[i,priceclose] > takeprofit ) or i == len(predictions)-1:
            testcount += 1
            #if ((prediction[i,2] > prediction[i,1] and prediction[i,2] > prediction[i,0])):
            #if round((prediction[0]-prediction[2])*100,2) > -7 or df[i,priceclose] < stoploss: 
            if True:
                money += ((money/100) * ((dfOneMin[i,priceclose]*(1 - fee*2) - buyprice) / buyprice)*100)
                print('Open Time:'+ time+  ' Sell Time:',dfOneMin[i,0]+ ' Change'+ '% 0.2f' % float(((float(dfOneMin[i,priceclose])*(1 - fee) - buyprice) / buyprice )*100) + ' sell:'+ '% 0.5f' % float(dfOneMin[i,priceclose]*(1 - fee)),' buy:'+ '% 0.5f' % buyprice+ " percent:"+ '% 0.2f' %  money)
                #f.write('Open Time:'+ time+  ' Sell Time:'+df[i,0]+ ' Change'+ '% 0.2f' % float(((float(df[i,priceclose])*(1 - fee) - buyprice) / buyprice )*100) + ' sell:'+ '% 0.5f' % float(df[i,priceclose]*(1 - fee))+' buy:'+ '% 0.5f' % buyprice+ " percent:"+ '% 0.2f' %  money+"\n")
                pos = -1
                time = dfOneMin[i,0]
                selltimes.append(time)
                #loopagain = True
                if buyprice > dfOneMin[i,priceclose]:
                    negtrade +=1
                else:
                    postrade +=1
    f.close

    
    print()
    print("Coin Change Over Time: "+ str((1-(dfOneMin[0,priceclose]/dfOneMin[len(dfOneMin)-1,priceclose]))*100))
    print()
    print("Signale:",str(signals)," Trades:", trades, " postive Trades:", postrade, " negative Trades:", negtrade, " Prozent:", money)
    print()
    print('##########################################################################')
    print()
    print()

    testfilename = coin+'_1M_'+testfromdate.strftime("%b%Y")+'_'+testtodate.strftime("%b%Y")

    #BacktraderTest(testfilename, buytimes, selltimes, testfromdate, testtodate)

    result =  "Signale: "+str(signals)+"     Trades: "+ str(trades)+ "     postive Trades: "+ str(postrade)+ "     negative Trades: "+ str(negtrade)+ "     Prozent: "+ str(money)   
    return [result,  [signals, trades, postrade, negtrade, money], predictions]

def multtestStrategie(filename, coins, testrange, predictions, tensor):

    dfs = []
    dir = "indicator_csv/"
    for coin in coins:
        
        testdata = dir+coin+"_5M_Jul2022_Dec2022.csv"
        df = pd.read_csv(testdata, header=0)
        df = np.array(df)
        df = np.delete( df, range(0, tensor) , 0)
        dfs.append(df)

    money = 100

    pos = -1
    buyprice = 0
    signals = 0
    trades=0
    postrade=0
    negtrade=0
    count = 0
    priceclose = 4
    loopagain=True
    fee= 0.001
    testcount = 0
    takeprofit = 0
    stoploss = 0
    percent = 0.02

    f = open(filename, 'w', newline='')
    print(str(len(predictions[0])))
    for i in range(0,len(predictions[0])):
        loopagain=True
        while loopagain:
            count +=1
            loopagain=False

            for index, prediction in enumerate(predictions):
                #if prediction[i,0] > prediction[i,1] and prediction[i,0] > prediction[i,2]:
                #print("Pred: "+str(round((prediction[i,0]-prediction[i,2])*100,2)))
                if round((prediction[i,0]-prediction[i,2])*100,2) > 10:
                    signals +=1
                    if pos == -1:
                        buyprice = dfs[0][i,priceclose]*(1)
                        takeprofit = buyprice * (1 + percent)
                        stoploss = buyprice * (1 - percent)
                        trades +=1
                        pos = index
                        time = dfs[index][i,0]
                        count = 1
                    else:
                        if dfs[index][i,priceclose] > buyprice:
                            pass
                            #takeprofit = dfs[index][i,priceclose] * 1.02
                            #stoploss = dfs[index][i,priceclose] * 0.98

            for index, prediction in enumerate(predictions):
                if pos == index and dfs[index][i,priceclose] < stoploss and round((prediction[i,0]-prediction[i,2])*100,2) > 10:
                    takeprofit = dfs[index][i,priceclose] * (1 + percent)
                    stoploss = dfs[index][i,priceclose] * (1 - percent)
                #if pos == index and count > testrange*4:-4
                if pos == index and (dfs[index][i,priceclose] > takeprofit  or i == len(prediction[0])-1):
                #if pos == index and (dfs[index][i,priceclose] > takeprofit or dfs[index][i,priceclose] < stoploss or i == len(prediction[0])-1):
                        testcount += 1
                    #if ((prediction[i,2] > prediction[i,1] and prediction[i,2] > prediction[i,0])):
                        money += ((money/100) * ((dfs[index][i,priceclose]*(1 - fee*2) - buyprice) / buyprice)*100)
                        #print('Open Time:'+ time+  ' Sell Time:',dfs[index][i,0]+ ' Change'+ '% 0.2f' % float(((float(dfs[index][i,priceclose])*(1 - fee) - buyprice) / buyprice )*100) + ' sell:'+ '% 0.5f' % float(dfs[index][i,priceclose]*(1 - fee)),' buy:'+ '% 0.5f' % buyprice+ " percent:"+ '% 0.2f' %  money)
                        f.write('Open Time:'+ time+  ' Sell Time:'+dfs[index][i,0]+ ' Change'+ '% 0.2f' % float(((float(dfs[index][i,priceclose])*(1 - fee) - buyprice) / buyprice )*100) + ' sell:'+ '% 0.5f' % float(dfs[index][i,priceclose]*(1 - fee))+' buy:'+ '% 0.5f' % buyprice+ " percent:"+ '% 0.2f' %  money+"\n")
                        pos = -1
                        #loopagain = True
                        if buyprice > dfs[index][i,priceclose]:
                            negtrade +=1
                        else:
                            postrade +=1
    f.close


    print("Testcount: "+str(testcount))
    print()
    print("Signale:",str(signals)," Trades:", trades, " postive Trades:", postrade, " negative Trades:", negtrade, " Prozent:", money)
    print()
    print('##########################################################################')
    print()
    print()

    result =  "Signale: "+str(signals)+"     Trades: "+ str(trades)+ "     postive Trades: "+ str(postrade)+ "     negative Trades: "+ str(negtrade)+ "     Prozent: "+ str(money)   
    return [result,  [signals, trades, postrade, negtrade, money], predictions]




