import numpy as np
import pandas as pd
from os.path import exists

from keras import models
from keras import Sequential
from keras import layers
from keras import optimizers
from keras import regularizers
#from keras_self_attention import SeqSelfAttention

priceclose = 4
pricelow = 3
pricehigh = 2
atr = 5
atrscale = 3
percent = 5

def neuralnet(filename, validationdata, LSTMTensor, Epochs, takeprof, testrange, Startpoints):

    Endpoints = 3
    dir = "indicator_csv/"
    df = pd.read_csv(dir+filename+'.csv', header=0)
    df = df.to_numpy()
    df = df[:, Startpoints]
    Startpoints = df.shape[1]


    save_model='neural_networks/'+filename

    #if not(exists('neural_networks/'+savemodel+'.csv')):
    if True:
        model = Sequential()

        model.add(layers.LSTM(
            16,
            activation='sigmoid',
            recurrent_activation='sigmoid',
            use_bias=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            return_sequences=False,
        ))

        model.add(layers.Dense(
            Endpoints, 
            activation="softmax", #linear
            use_bias= True, 
            name="end"))  

        model.compile(optimizer=optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy')

    if False:

        model = Sequential()

        #model.add(layers.Dropout(0.4))

        model.add(layers.LSTM(
            16,
            activation='sigmoid',
            recurrent_activation='sigmoid',
            use_bias=True,
            #kernel_initializer='glorot_uniform',
            #recurrent_initializer='orthogonal',
            #bias_initializer='zeros',
            #unit_forget_bias=True,
            #kernel_regularizer=None,
            #recurrent_regularizer=None,
            #bias_regularizer=None,
            #activity_regularizer=None,
            #kernel_constraint=None,
            #recurrent_constraint=None,
            #bias_constraint=None,
            dropout=0.2,
            recurrent_dropout=0.2,
            return_sequences=True,
            #return_state=False,
            #go_backwards=False,
            #stateful=False,
            #time_major=False,
            #unroll=False
        ))

        model.add(layers.Dropout(0.2))

        model.add(layers.LSTM(
            16,
            activation='sigmoid',
            recurrent_activation='sigmoid',
            use_bias=True,
            dropout = 0.2,
            recurrent_dropout=0.2,
            return_sequences=False,
        ))

        #model.add(layers.Dropout(0.4))

        model.add(layers.Dense(
                        64 , 
                        activation="sigmoid", #"linear"
                        # kernel_initializer='ones',  
                        # kernel_regularizer= regularizers.L1(0.01),
                        # bias_regularizer= regularizers.L2(0.01),
                        # activity_regularizer= regularizers.L1(0.04),
                         use_bias = True, 
        #                 name="layer3"
        ))

    
        # model.add(layers.LSTM(
        #     64,
        #     activation='sigmoid',
        #     recurrent_activation='sigmoid',
        #     #use_bias=True,
        #     #kernel_initializer='glorot_uniform',
        #     #recurrent_initializer='orthogonal',
        #     #bias_initializer='zeros',
        #     #unit_forget_bias=True,
        #     #kernel_regularizer=None,
        #     #recurrent_regularizer=None,
        #     #bias_regularizer=None,
        #     #activity_regularizer=None,
        #     #kernel_constraint=None,
        #     #recurrent_constraint=None,
        #     #bias_constraint=None,
        #     #dropout=0.2,
        #     #recurrent_dropout=0.0,
        #     #return_sequences=False,
        #     #return_state=False,
        #     #go_backwards=False,
        #     #stateful=False,
        #     #time_major=False,
        #     #unroll=False
        # ))

        # model.add(layers.Dense(
        #     16,
        #     activation='sigmoid',
        # #     #use_bias=True,
        # #     #kernel_initializer='glorot_uniform',
        # #     #bias_initializer='zeros',
        # #     #kernel_regularizer=None,
        # #     #bias_regularizer=None,
        # #     #activity_regularizer=None,
        # #     #kernel_constraint=None,
        # #     #bias_constraint=None,
        # ))

        model.add(layers.Dense(
            Endpoints, 
            activation="softmax", #linear
            use_bias= True, 
            name="end"))   

        #model.add(layers.Dense(1, activation="linear", name="end"))

        #model.compile(optimizer=optimizers.Adam(learning_rate=0.01),loss="mse")
        #model.compile(optimizer=optimizers.RMSprop(learning_rate=0.01),loss="mse")
        model.compile(optimizer=optimizers.Adam(learning_rate=0.01),loss='categorical_crossentropy')

    else:
        pass
        #model = models.load_model(save_model)


    print('')
    print('#######################################################')
    print('')
    print('Train Neural Network')
    print('')     
    
    X = []  #Trainingsdaten
    Y = []  #Validierungsdaten

    Xtest = []  #Test Trainingsdaten
    Ytest = []  #Test Validierungsdaten

    traincut = len(df)*0.8

    for i in range(0, len(validationdata)):
        if i > traincut-1:
            Xtest.append(np.array(df[i]))
        else:
            X.append(np.array(df[i]))

    for i in range(0, len(validationdata)):
        if i > traincut-1:
            Ytest.append(np.array(validationdata[i]))
        else:
            Y.append(np.array(validationdata[i]))

    X = np.array(X).reshape(-1, Startpoints)
    Y = np.array(Y).reshape(-1, Endpoints)
    Xtest = np.array(Xtest).reshape(-1, Startpoints)
    Ytest = np.array(Ytest).reshape(-1, Endpoints)

    X = np.asarray(X).astype(np.float32)
    Xtest = np.asarray(Xtest).astype(np.float32)

    X3D = []

    for i in range(LSTMTensor, len(X)):
        X3D.append(np.array(X[i-LSTMTensor:i]))

    X3D = np.asarray(X3D).reshape(-1, LSTMTensor, Startpoints)

    X3Dtest = []

    for i in range(LSTMTensor, len(Xtest)):
        X3Dtest.append(np.array(Xtest[i-LSTMTensor:i]))

    X3Dtest = np.asarray(X3Dtest).reshape(-1, LSTMTensor, Startpoints)

    Y3D = []

    Y3D = np.delete( Y, range(0, LSTMTensor), axis = 0)

    Y3Dtest = []

    Y3Dtest = np.delete( Ytest, range(0, LSTMTensor), axis = 0)

    model.fit(X3D, Y3D, batch_size = len(X3D)//100, epochs=Epochs) #batch_size=,
    #model.fit(X3D, Y3D, batch_size = 64, epochs=Epochs) #batch_size=,

    modeleval = model.evaluate(X3Dtest, Y3Dtest)

    predictions = model.predict(X3D)
    predictions = predictions.reshape(-1,Endpoints)


    csvpred = pd.DataFrame(predictions)
    csvpred.to_csv('pred.csv')

    # prediction_single = model.predict([[-0.0009203025028212422,0.01346693997061647,0.016615723310162766,-0.004102174024686075,0.015054721752398153,0.03082965635822733,-0.01073755171748349,-0.003000271028160521,0.030396761758821497,-220.65864734249772,-350.89438157284206,78.14145642377979,76.47526705669145,48.668410772362456,-22.99015312091231,-0.004102174024686075,0.032793917011594426,-0.04099826506096658,657.3759936693134,23.610677572067328]])

    # print(prediction_single)


    #print(model.layers[0].weights)

    #print(model.layers[1].weights)

    #print(model.layers[2].weights)

    #print(model.layers[3].weights)

    #print(model.summary())

    model.save(save_model)  

    return modeleval

def nolstm(filename, validationdata, LSTMTensor, Epochs, takeprof, testrange):

    Startpoints = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] 
    Endpoints = 3
    df = pd.read_csv('indicator_csv/'+filename+'.csv', header=0)
    df = df.to_numpy()
    df = df[:, Startpoints]
    Startpoints = df.shape[1]

    save_model='neural_networks/'+filename

    model = Sequential()

    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(
        64,
        activation='sigmoid',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    ))

    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(
        64,
        activation='sigmoid',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    ))

    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(
        64,
        activation='sigmoid',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    ))

    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(
        64,
        activation='sigmoid',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    ))

    model.add(layers.Dense(Endpoints, activation="softmax", use_bias= True, name="end"))   #linear

    #model.compile(optimizer=optimizers.Adam(learning_rate=0.01),loss="mse")
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.01),loss="mse")


    print('')
    print('#######################################################')
    print('')
    print('Train Neural Network')
    print('')     
    
    X = []  #Trainingsdaten
    Y = []  #Validierungsdaten

    Xtest = []  #Test Trainingsdaten
    Ytest = []  #Test Validierungsdaten

    traincut = len(df)*0.8

    for i in range(0, len(validationdata)):
        if i > traincut-1:
            Xtest.append(np.array(df[i]))
        else:
            X.append(np.array(df[i]))

    for i in range(0, len(validationdata)):
        if i > traincut-1:
            Ytest.append(np.array(validationdata[i]))
        else:
            Y.append(np.array(validationdata[i]))

    X = np.array(X).reshape(-1, Startpoints)
    Y = np.array(Y).reshape(-1, Endpoints)
    Xtest = np.array(Xtest).reshape(-1, Startpoints)
    Ytest = np.array(Ytest).reshape(-1, Endpoints)

    X = np.asarray(X).astype(np.float32)
    Xtest = np.asarray(Xtest).astype(np.float32)

    
    model.fit(X, Y, batch_size = len(X)//100, epochs=Epochs) #batch_size=,
    #model.fit(X3D, Y3D, batch_size = 64, epochs=Epochs) #batch_size=,

    modeleval = model.evaluate(Xtest, Ytest)

    predictions = model.predict(X)
    predictions = predictions.reshape(-1,Endpoints)


    csvpred = pd.DataFrame(predictions)
    csvpred.to_csv('pred.csv')

    # prediction_single = model.predict([[-0.0009203025028212422,0.01346693997061647,0.016615723310162766,-0.004102174024686075,0.015054721752398153,0.03082965635822733,-0.01073755171748349,-0.003000271028160521,0.030396761758821497,-220.65864734249772,-350.89438157284206,78.14145642377979,76.47526705669145,48.668410772362456,-22.99015312091231,-0.004102174024686075,0.032793917011594426,-0.04099826506096658,657.3759936693134,23.610677572067328]])

    # print(prediction_single)


    #print(model.layers[0].weights)

    #print(model.layers[1].weights)

    #print(model.layers[2].weights)

    #print(model.layers[3].weights)

    #print(model.summary())

    model.save(save_model)  

    return modeleval