import backtrader as bt
import csv


class allindicators(bt.Strategy):
    params = (
        ('scoin', 'BTC'),
        ('time', '15M'),
        ('start', 'Jan2021'),
        ('end', 'Jan2021'),
        ('test', False),

        #Trend
        ('emafast', 20),
        ('emaneutral', 50),
        ('emaslow', 200),
        ('smafast', 20),
        ('smaneutral', 50),
        ('smaslow', 200),
        ('demafast', 20),
        ('demaneutral', 50),
        ('demaslow', 200),
        ('rsi', 14),
        ('macdfast', 12),
        ('macdslow', 26),
        ('macdsmooth', 9),

        #Momentum
        ('stochperiod', 14),
        ('stochdfast', 3),
        ('stochdslow', 3),
        ('rsi', 14),
        ('will', 14),
        ('momosc', 30),

        #Volatility
        ('bollinger', 20),
        ('standarddev', 20),
        ('atr', 14),
        ('adx', 14),

        #Ichimoku
        ('tenkan', 9),
        ('kijun', 26),
        ('senkou', 52),
        ('senkou_lead', 26),
        ('chikou', 0),

        ('printlog', False),
    )

    def __init__(self):
    
        self.time = self.data.datetime
        self.pricelow = self.data.low
        self.pricehigh = self.data.high
        self.priceopen = self.data.open
        self.priceclose = self.data.close

        #Trend Indicator
        self.emafast = bt.indicators.ExponentialMovingAverage(period=self.params.emafast)
        self.emaneutral = bt.indicators.ExponentialMovingAverage(period=self.params.emaneutral)
        self.emaslow = bt.indicators.ExponentialMovingAverage(period=self.params.emaslow)
        self.smafast = bt.indicators.MovingAverageSimple(period=self.params.smafast)
        self.smaneutral = bt.indicators.MovingAverageSimple(period=self.params.smaneutral)
        self.smaslow = bt.indicators.MovingAverageSimple(period=self.params.smaslow)
        self.demafast = bt.indicators.DoubleExponentialMovingAverage(period=self.params.demafast)
        self.demaneutral = bt.indicators.DoubleExponentialMovingAverage(period=self.params.demaneutral)
        self.demaslow = bt.indicators.DoubleExponentialMovingAverage(period=self.params.demaslow)
        self.macd = bt.indicators.MACD(period_me1 = self.params.macdfast, period_me2 = self.params.macdslow, period_signal = self.params.macdsmooth)

        #Momentum
        self.stochastic = bt.indicators.Stochastic(period = self.params.stochperiod, period_dfast = self.params.stochdfast, period_dslow = self.params.stochdslow)
        self.rsi = bt.indicators.RSI(period = self.params.rsi)
        self.willR = bt.indicators.WilliamsR(period = self.params.will)
        self.momosz = bt.indicators.MomentumOscillator(period = self.params.momosc)

        #Volatility
        self.bollinger = bt.indicators.BollingerBands(period = self.params.bollinger)
        self.standarddev = bt.indicators.StandardDeviation(period = self.params.standarddev)
        self.atr = bt.indicators.AverageTrueRange(period = self.params.atr)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(period = self.params.adx)
        
        self.ichimoku = bt.indicators.Ichimoku(tenkan = self.params.tenkan, kijun = self.params.kijun, senkou = self.params.senkou, senkou_lead = self.params.senkou_lead, chikou = self.params.chikou)

        if self.params.test:
            self.csv = 'test'+self.params.scoin+'_'+self.params.time+'.csv'
        else:
            self.csv = 'indicator_csv/'+self.params.scoin+'_'+self.params.time+'_'+self.params.start+'_'+self.params.end+'.csv'

        # self.row = ['time','priceopen', 'pricehigh', 'pricelow', 'priceclose', 'atr', 'emafast', 'emaneutral', 'emaslow', 'macd', 'macdsignal',
        #     'stochasticK', 'stochasticD', 'rsi', 'adx', 'emadiffn', 'emadifns', ' macddif', 'stochdif']

        self.row = ['time','priceopen', 'pricehigh', 'pricelow', 'priceclose', 'atr', 'momosz', 'emafast', 'emaneutral', 'emaslow', 'smafast', 'smaneutral', 'smaslow', 'demafast', 'demaneutral', 'demaslow', 'macd', 'macdsignal',
            'stochasticK', 'stochasticD', 'rsi', 'willR', 'bollingermid', 'bollingertop', 'bollingerbot', 'standarddev', 'adx', 'emadiffn', 'emadifns', 'macddif', 'stochdif', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']

        self.f = open(self.csv, 'w', newline='')

        self.writer = csv.writer(self.f)

        self.writer.writerow(self.row)

    def next(self):
        #Trend
        emafast = (self.emafast.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        emaneutral = (self.emaneutral.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        emaslow = (self.emaslow.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        emadiffn = emafast - emaneutral
        emadifns = emaneutral - emafast
        smafast = (self.smafast.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        smaneutral = (self.smaneutral.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        smaslow = (self.smaslow.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        demafast = (self.demafast.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        demaneutral = (self.demaneutral.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        demaslow = (self.demaslow.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        macd = self.macd.lines[0][0]
        macdsignal = self.macd.lines[1][0]
        macddif = macd - macdsignal

        #Momentum
        stochasticK = self.stochastic.lines[0][0]
        stochasticD = self.stochastic.lines[1][0]
        stochdif = stochasticK - stochasticD
        rsi = self.rsi.lines[0][0]
        willR = self.willR.lines[0][0]
        momosc = self.momosz.lines[0][0]


        #Volatility
        bollingermid = (self.bollinger.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        bollingertop = (self.bollinger.lines[1][0] - self.priceclose[0]) / self.priceclose[0]
        bollingerbot = (self.bollinger.lines[2][0] - self.priceclose[0]) / self.priceclose[0]
        standarddev = self.standarddev.lines[0][0]
        adx = self.adx.lines[0][0]
        atr = self.atr.lines[0][0]

        tenkan = (self.ichimoku.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        kijun = (self.ichimoku.lines[1][0] - self.priceclose[0])  / self.priceclose[0]
        senkoua = (self.ichimoku.lines[2][0] - self.priceclose[0])  / self.priceclose[0]
        senkoub = (self.ichimoku.lines[3][0] - self.priceclose[0])  / self.priceclose[0]
        chikou = (self.ichimoku.lines[4][0] - self.priceclose[0])  / self.priceclose[0]


        time = bt.num2date(self.time[0])


        # self.row = [time, self.priceopen[0], self.pricehigh[0], self.pricelow[0], self.priceclose[0], atr, emafast, emaneutral, emaslow, macd, macdsignal,
        #     stochasticK, stochasticD, rsi, adx, emadiffn, emadifns, macddif, stochdif]

        self.row = [time, self.priceopen[0], self.pricehigh[0], self.pricelow[0], self.priceclose[0], atr, momosc, emafast, emaneutral, emaslow, smafast, smaneutral, smaslow, demafast, demaneutral, demaslow, macd, macdsignal,
            stochasticK, stochasticD, rsi, willR, bollingermid, bollingertop, bollingerbot, standarddev, adx, emadiffn, emadifns, macddif, stochdif, tenkan, kijun, senkoua, senkoub, chikou]



        self.writer.writerow(self.row)

    def stop(self):
        self.f.close()



class ichiindicator(bt.Strategy):
    params = (
        ('scoin', 'BTC'),
        ('time', '15M'),
        ('test', False),

        ('tenkan', 9),
        ('kijun', 26),
        ('senkou', 52),
        ('senkou_lead', 26),
        ('chikou', 26),

        ('printlog', False),
    )

    def __init__(self):
        self.time = self.data.datetime
        self.pricelow = self.data.low
        self.pricehigh = self.data.high
        self.priceopen = self.data.open
        self.priceclose = self.data.close

        self.ichimoku = bt.indicators.Ichimoku(tenkan = self.params.tenkan, kijun = self.params.kijun, senkou = self.params.senkou, senkou_lead = self.params.senkou_lead, chikou = self.params.chikou)


        if self.params.test:
            self.csv = 'test_ichi_'+self.params.scoin+'_'+self.params.time+'.csv'
        else:
            self.csv = 'ichi_'+self.params.scoin+'_'+self.params.time+'.csv'

        self.row = ['time','priceopen', 'pricehigh', 'pricelow', 'priceclose', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']

        self.f = open(self.csv, 'w', newline='')

        self.writer = csv.writer(self.f)

        self.writer.writerow(self.row)


    def next(self):

        tenkan = (self.ichimoku.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        kijun = (self.ichimoku.lines[1][0] - self.priceclose[0])  / self.priceclose[0]
        senkoua = (self.ichimoku.lines[2][0] - self.priceclose[0])  / self.priceclose[0]
        senkoub = (self.ichimoku.lines[3][0] - self.priceclose[0])  / self.priceclose[0]
        chikou = (self.ichimoku.lines[4][0] - self.priceclose[0])  / self.priceclose[0]



        time = bt.num2date(self.time[0])


        self.row = [time, self.priceopen[0], self.pricehigh[0], self.pricelow[0], self.priceclose[0], tenkan, kijun, senkoua, senkoub, chikou]


        self.writer.writerow(self.row)
        
    def stop(self):
        self.f.close()


class relativeindicator(bt.Strategy):

    params = (
        ('scoin', 'BTC'),
        ('time', '15M'),
        ('filename', 'filename'),
        ('test', False),

        #Trend
        ('emafast', 20),
        ('emaneutral', 50),
        ('emaslow', 200),
        ('smafast', 20),
        ('smaneutral', 50),
        ('smaslow', 200),
        ('demafast', 20),
        ('demaneutral', 50),
        ('demaslow', 200),
        ('rsi', 14),
        ('macdfast', 12),
        ('macdslow', 26),
        ('macdsmooth', 9),

        #Momentum
        ('stochperiod', 14),
        ('stochdfast', 3),
        ('stochdslow', 3),
        ('rsi', 14),
        ('will', 14),
        ('momosc', 30),

        #Volatility
        ('bollinger', 20),
        ('standarddev', 20),
        ('atr', 14),
        ('adx', 14),

        ('printlog', False),
    )


    def __init__(self):
    
        self.time = self.data.datetime
        self.pricelow = self.data.low
        self.pricehigh = self.data.high
        self.priceopen = self.data.open
        self.priceclose = self.data.close

        #Trend Indicator
        self.emafast = bt.indicators.ExponentialMovingAverage(period=self.params.emafast)
        self.emaneutral = bt.indicators.ExponentialMovingAverage(period=self.params.emaneutral)
        self.emaslow = bt.indicators.ExponentialMovingAverage(period=self.params.emaslow)
        self.smafast = bt.indicators.MovingAverageSimple(period=self.params.smafast)
        self.smaneutral = bt.indicators.MovingAverageSimple(period=self.params.smaneutral)
        self.smaslow = bt.indicators.MovingAverageSimple(period=self.params.smaslow)
        self.demafast = bt.indicators.DoubleExponentialMovingAverage(period=self.params.demafast)
        self.demaneutral = bt.indicators.DoubleExponentialMovingAverage(period=self.params.demaneutral)
        self.demaslow = bt.indicators.DoubleExponentialMovingAverage(period=self.params.demaslow)
        self.macd = bt.indicators.MACD(period_me1 = self.params.macdfast, period_me2 = self.params.macdslow, period_signal = self.params.macdsmooth)

        #Momentum
        self.stochastic = bt.indicators.Stochastic(period = self.params.stochperiod, period_dfast = self.params.stochdfast, period_dslow = self.params.stochdslow)
        self.rsi = bt.indicators.RSI(period = self.params.rsi)
        self.willR = bt.indicators.WilliamsR(period = self.params.will)
        self.momosz = bt.indicators.MomentumOscillator(period = self.params.momosc)

        #Volatility
        self.bollinger = bt.indicators.BollingerBands(period = self.params.bollinger)
        self.standarddev = bt.indicators.StandardDeviation(period = self.params.standarddev)
        self.atr = bt.indicators.AverageTrueRange(period = self.params.atr)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(period = self.params.adx)

        dir = "indicator_csv/"
        if self.params.test:
            self.csv = 'test'+self.params.scoin+'_'+self.params.time+'.csv'
        else:
            self.csv = dir+self.params.filename+'.csv'

        # self.row = ['time','priceopen', 'pricehigh', 'pricelow', 'priceclose', 'atr', 'emafast', 'emaneutral', 'emaslow', 'macd', 'macdsignal',
        #     'stochasticK', 'stochasticD', 'rsi', 'adx', 'emadiffn', 'emadifns', ' macddif', 'stochdif']

        # self.row = ['time','priceopen', 'pricehigh', 'pricelow', 'priceclose', 'emadiffn', 'emadifns', 'macddif', 'stochdif', 'rsi', 'willR', 'momosc',
        #   'bollingermid', 'bollingertop', 'bollingerbot', 'standarddev', 'adx', 'atr']

        self.f = open(self.csv, 'w', newline='')

        self.writer = csv.writer(self.f)

    def next(self):
        #Trend
        emafast = (self.emafast.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        emaneutral = (self.emaneutral.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        emaslow = (self.emaslow.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        emadiffn = emafast - emaneutral     #this
        emadifns = emaneutral - emaslow     #this

        macd = self.macd.lines[0][0]
        macdsignal = self.macd.lines[1][0]
        macddif = macd - macdsignal         #this

        #Momentum
        stochasticK = self.stochastic.lines[0][0]
        stochasticD = self.stochastic.lines[1][0]
        stochdif = stochasticK - stochasticD    #this
        rsi = self.rsi.lines[0][0]              #this
        willR = self.willR.lines[0][0]          #this
        momosc = self.momosz.lines[0][0]        #this


        #Volatility
        bollingermid = (self.bollinger.lines[0][0] - self.priceclose[0]) / self.priceclose[0]   #this
        bollingertop = (self.bollinger.lines[1][0] - self.priceclose[0]) / self.priceclose[0]   #this
        bollingerbot = (self.bollinger.lines[2][0] - self.priceclose[0]) / self.priceclose[0]   #this
        standarddev = self.standarddev.lines[0][0]
        adx = self.adx.lines[0][0]
        atr = self.atr.lines[0][0]

        time = bt.num2date(self.time[0])
        #time= self.time[0]


        self.row = [time, self.priceopen[0], self.pricehigh[0], self.pricelow[0], self.priceclose[0], emadiffn, emadifns, macddif, stochdif, rsi, willR, momosc,
          bollingermid, bollingertop, bollingerbot, standarddev, adx, atr]

        self.writer.writerow(self.row)

    def stop(self):
        self.f.close()

class newindicator(bt.Strategy):

    params = (
        ('scoin', 'BTC'),
        ('time', '15M'),
        ('filename', 'filename'),
        ('test', False),

        #Trend
        ('emafast', 20),
        ('emaneutral', 50),
        ('emaslow', 200),
        ('smafast', 20),
        ('smaneutral', 50),
        ('smaslow', 200),
        ('demafast', 20),
        ('demaneutral', 50),
        ('demaslow', 200),
        ('rsi', 14),
        ('macdfast', 12),
        ('macdslow', 26),
        ('macdsmooth', 9),

        #Momentum
        ('stochperiod', 14),
        ('stochdfast', 3),
        ('stochdslow', 3),
        ('rsi', 14),
        ('will', 14),
        ('momosc', 30),

        #Volatility
        ('bollinger', 20),
        ('standarddev', 20),
        ('atr', 14),
        ('adx', 14),

        ('printlog', False),
    )


    def __init__(self):

        if self.params.time == "1M":
            mult = 5
        else:
            mult = 1
    
        self.time = self.data.datetime
        self.pricelow = self.data.low
        self.pricehigh = self.data.high
        self.priceopen = self.data.open
        self.priceclose = self.data.close
        self.volume =self.data.volume

        #Trend Indicator
        self.emafast = bt.indicators.ExponentialMovingAverage(period=self.params.emafast*mult)
        self.emaneutral = bt.indicators.ExponentialMovingAverage(period=self.params.emaneutral*mult)
        self.emaslow = bt.indicators.ExponentialMovingAverage(period=self.params.emaslow*mult)
        self.smafast = bt.indicators.MovingAverageSimple(period=self.params.smafast*mult)
        self.smaneutral = bt.indicators.MovingAverageSimple(period=self.params.smaneutral*mult)
        self.smaslow = bt.indicators.MovingAverageSimple(period=self.params.smaslow*mult)
        self.demafast = bt.indicators.DoubleExponentialMovingAverage(period=self.params.demafast*mult)
        self.demaneutral = bt.indicators.DoubleExponentialMovingAverage(period=self.params.demaneutral*mult)
        self.demaslow = bt.indicators.DoubleExponentialMovingAverage(period=self.params.demaslow*mult)
        self.macd = bt.indicators.MACD(period_me1 = self.params.macdfast*mult, period_me2 = self.params.macdslow*mult, period_signal = self.params.macdsmooth*mult)

        #Momentum
        self.stochastic = bt.indicators.Stochastic(period = self.params.stochperiod*mult, period_dfast = self.params.stochdfast*mult, period_dslow = self.params.stochdslow*mult)
        self.rsi = bt.indicators.RSI(period = self.params.rsi*mult)
        self.willR = bt.indicators.WilliamsR(period = self.params.will*mult)
        self.momosz = bt.indicators.MomentumOscillator(period = self.params.momosc*mult)

        #Volatility
        self.bollinger = bt.indicators.BollingerBands(period = self.params.bollinger*mult)
        self.standarddev = bt.indicators.StandardDeviation(period = self.params.standarddev*mult)
        self.atr = bt.indicators.AverageTrueRange(period = self.params.atr*mult)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(period = self.params.adx*mult)

        dir = "indicator_csv/"
        if self.params.test:
            self.csv = 'test'+self.params.scoin+'_'+self.params.time+'.csv'
        else:
            self.csv = dir+self.params.filename+'.csv'

        self.f = open(self.csv, 'w', newline='')

        self.writer = csv.writer(self.f)

    def next(self):
        #Trend
        emafast = (self.emafast.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        emaneutral = (self.emaneutral.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        emaslow = (self.emaslow.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        emadiffn = emafast - emaneutral     #this
        emadifns = emaneutral - emaslow     #this

        macd = self.macd.lines[0][0]
        macdsignal = self.macd.lines[1][0]
        macddif = macd - macdsignal         #this

        #Momentum
        stochasticK = self.stochastic.lines[0][0]
        stochasticD = self.stochastic.lines[1][0]
        stochdif = stochasticK - stochasticD    #this
        rsi = self.rsi.lines[0][0]              #this
        willR = self.willR.lines[0][0]          #this
        momosc = self.momosz.lines[0][0]        #this


        #Volatility
        bollingermid = (self.bollinger.lines[0][0] - self.priceclose[0]) / self.priceclose[0]   #this
        bollingertop = (self.bollinger.lines[1][0] - self.priceclose[0]) / self.priceclose[0]   #this
        bollingerbot = (self.bollinger.lines[2][0] - self.priceclose[0]) / self.priceclose[0]   #this
        standarddev = self.standarddev.lines[0][0]
        adx = self.adx.lines[0][0]
        atr = self.atr.lines[0][0]

        #Volume
        volume = self.volume[0]

        time = bt.num2date(self.time[0])
        #time= self.time[0]


        self.row = [time, self.priceopen[0], self.pricehigh[0], self.pricelow[0], self.priceclose[0], volume, emadiffn, emadifns, macd, macdsignal, macddif, stochdif, rsi, willR, momosc,
          bollingermid, bollingertop, bollingerbot, standarddev, adx, atr]

        self.writer.writerow(self.row)

    def stop(self):
        self.f.close()


class CHATGPT(bt.Strategy):

    params = (
        ('scoin', 'BTC'),
        ('time', '15M'),
        ('filename', 'filename'),
        ('test', False),

        ('emafast', 20),
        ('emaneutral', 50),
        ('emaslow', 200),
        ('bollinger', 20),
        ('rsi', 14),
        ('macdfast', 12),
        ('macdslow', 26),
        ('macdsmooth', 9),

        ("tenkan_period", 9),
        ("kijun_period", 26),
        ("senkou_b_period", 52),
        ("chikou_period", 26),        

        ('printlog', False),
    )


    def __init__(self):
    
        self.time = self.data.datetime
        self.pricelow = self.data.low
        self.pricehigh = self.data.high
        self.priceopen = self.data.open
        self.priceclose = self.data.close
        self.volume =self.data.volume

        #Trend Indicator
        self.emafast = bt.indicators.ExponentialMovingAverage(period=self.params.emafast)
        self.emaneutral = bt.indicators.ExponentialMovingAverage(period=self.params.emaneutral)
        self.emaslow = bt.indicators.ExponentialMovingAverage(period=self.params.emaslow)
        self.bollinger = bt.indicators.BollingerBands(period = self.params.bollinger)
        self.rsi = bt.indicators.RSI(period = self.params.rsi)
        self.macd = bt.indicators.MACD(period_me1 = self.params.macdfast, period_me2 = self.params.macdslow, period_signal = self.params.macdsmooth)
        self.ichi = bt.indicators.Ichimoku(self.data, tenkan=self.p.tenkan_period, kijun=self.p.kijun_period, senkou = self.p.senkou_b_period,  chikou = self.p.chikou_period)

        if self.params.test:
            self.csv = 'test'+self.params.scoin+'_'+self.params.time+'.csv'
        else:
            self.csv = 'indicator_csv/'+self.params.filename+'.csv'

        self.f = open(self.csv, 'w', newline='')

        self.writer = csv.writer(self.f)

    def next(self):
        #Trend
        emafast = (self.emafast.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        emaneutral = (self.emaneutral.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        emaslow = (self.emaslow.lines[0][0] - self.priceclose[0]) / self.priceclose[0]
        emadiffn = emafast - emaneutral     #this
        emadifns = emaneutral - emaslow     #this
        bollingermid = (self.bollinger.lines[0][0] - self.priceclose[0]) / self.priceclose[0]   #this
        bollingertop = (self.bollinger.lines[1][0] - self.priceclose[0]) / self.priceclose[0]   #this
        bollingerbot = (self.bollinger.lines[2][0] - self.priceclose[0]) / self.priceclose[0]   #this
        rsi = self.rsi.lines[0][0]              #this
        macd = self.macd.lines[0][0]
        macdsignal = self.macd.lines[1][0]
        tenkan = self.ichi.lines[0][0]
        kijun = self.ichi.lines[1][0]
        senkou_b = self.ichi.lines[2][0]
        chikou = self.ichi.lines[3][0]



        time = bt.num2date(self.time[0])
        #time= self.time[0]


        self.row = [time, self.priceopen[0], self.pricehigh[0], self.pricelow[0], self.priceclose[0], emadiffn, emadifns, bollingermid, bollingertop, bollingerbot, rsi, macd, macdsignal, 
            tenkan, kijun, senkou_b, chikou]

        self.writer.writerow(self.row)

    def stop(self):
        self.f.close()