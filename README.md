# Bachelorarbeit

In der requirements.txt müssten alle benötigten Bibliotheken stehen, um die neuronalen Netze zu testen.
Die Daten hab ich schon von Binance heruntergeladen und aufbereitet.

Zum "schnellen" testen des Programms muss man starthere.py ausführen.

Die Einstellungen wurden so gewählt, dass die Netze möglichst schnell berechnet werden.
Die Einstellungen wurden so nicht in der bachelorarbeit benutzt

Unter der "main" Klasse kann man Einstellungen wie gewünscht ändern.


    coins = ["BTC","ETH","XRP", "BNB", "DOGE"]
    #coins = ["BTC"]    <- wenn man eine bestimmte Coin testen möchte

    candlesizes = ["5M","15M","30M","1H"]
    #candlesizes = ["5M"]     <- wenn man ein bestimmten Zeitinterval testen möchte

    tensors=[12, 24, 48]      <- so in der BA
    tensors=[4]         <- Tensorgröße

    testranges = [12, 24, 48]     <- so in der BA
    testranges = [12]      <- Die Candleanzahl die in die Zukunft geschaut wird, um zu bestimmen ob eine Candle gut oder schlecht ist

    stds = [0.1 , 0.3 , 0.5]    <- so in der BA
    stds = [0.1]         <- Standardabweichung für den Gewinn oder Verlust der innerhalb der testrange erreicht werden muss, damit die Candle entsprechend gewertet wird

    Epochs =   3        <- Epochen des NN
    wdh = 1               <- erstellt mehrere netze mit den selben Einstellungen, um einen Mittelwert zu bilden
