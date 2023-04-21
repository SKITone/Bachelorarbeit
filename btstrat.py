import backtrader as bt

class timebuy(bt.Strategy):
    params = (
        ('buytimes', ''),
        ('selltimes', '')

    )
    def __init__(self):
        self.time = self.data.datetime
        self.price = self.data.close
        self.buytimes = self.params.buytimes
        self.selltimes = self.params.selltimes


    def next(self):
        time = str(bt.num2date(self.time[0]))

        for i in range(len(self.selltimes)):
            if self.selltimes[i] == time:
                self.sell()

        for i in range(len(self.buytimes)):
            if self.buytimes[i] == time:
                self.buy()