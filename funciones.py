import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cycler

import nltk
from wordcloud import WordCloud #
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


import streamlit as st
import MetaTrader5 as mt5
from datetime import datetime
import ta

colors = cycler('color',
                ['#669FEE', '#66EE91', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
plt.rc('figure', facecolor='#313233')
plt.rc('axes', facecolor="#313233", edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors,
       labelcolor='gray')
plt.rc('grid', color='474A4A', linestyle='solid')
plt.rc('xtick', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('legend', facecolor="#313233", edgecolor="#313233")
plt.rc("text", color="#C9C9C9")
plt.rc('figure', facecolor='#313233')

def missing_values_table(df):
    '''
    Se pasa como parametro un df de pandas, devuelve la 
    cantidad de valores NaN y a que porcentaje del total de valores corresponen
    
    '''
    
    mis_val = df.isna().sum()
    mis_val_percent = 100 * df.isna().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    return mis_val_table_ren_columns

##################### ANALISIS DE ESNTIMIENTO
def generate_wordcloud(texts):
    # Get English prepositions
    stop_words = set(stopwords.words('english'))

    # Combine all the texts into a single string
    combined_text = ' '.join(str(text) for text in texts)

    # Tokenize the combined text into words
    words = word_tokenize(combined_text)

    # Filter out the prepositions
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]

    # Create a frequency distribution of the words
    freq_dist = nltk.FreqDist(filtered_words)

    # Create the word cloud with word frequencies as input
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dist)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def generate_frec(texts,orden:bool):
    combined_text = ' '.join(str(text) for text in texts)
    words = word_tokenize(combined_text)
    filtered_words = [word.lower() for word in words ]
    freq_dist = nltk.FreqDist(filtered_words)
    # pasamos a dataframe
    df_frec=pd.DataFrame(list(freq_dist.items()),columns=['Word','Frecuencia'])
    # Ordenamos
    df_frec.sort_values('Frecuencia',ascending=orden,inplace=True)
    # reiniciamos indice
    df_frec.reset_index(drop=True,inplace=True)
    # Graficamos
    return df_frec

################ DIVERSIFICAR CARTERA
# FXPRO
server="FxPro-MT5"
login = "5843847"
password="bYsJsFu9"
if  mt5.initialize(server=server, login=login, password=password)==False:
    mt5.initialize(server=server, login=login, password=password)
    print("Se conecto")

def preprocessing_mt5(symbol, n, timeframe=mt5.TIMEFRAME_H1):

    if mt5.initialize() == False:
        mt5.initialize()
        
    # Current date extract
    utc_from = datetime.now()
    # Import the data into a tuple
    rates = mt5.copy_rates_from(symbol, timeframe, utc_from, n)
    # Tuple to dataframe
    rates_frame = pd.DataFrame(rates)
    # Convert time in seconds into the datetime format
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
    # Convert the column "time" in the right format
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], format='%Y-%m-%d')
    # Set column time as the index of the dataframe
    rates_frame = rates_frame.set_index('time')

    # Delete the two last columns
    rates_frame = rates_frame.iloc[:,:-2]
    # Rename
    rates_frame.columns = ["open", "high", "low", "close", "volume"]
    
    # Desconectar del servidor de MetaTrader 5
    mt5.shutdown()
    return rates_frame

## Estrategia SMAy RSI


def support_resistance(input,n=1000,duration=5,spread=0, mt5_live=True):

    if mt5_live:
        df = preprocessing_mt5(input,n=n)
    
    # Support and resistance building
    df["support"] = np.nan
    df["resistance"] = np.nan

    df.loc[(df["low"].shift(5) > df["low"].shift(4)) &
            (df["low"].shift(4) > df["low"].shift(3)) &
            (df["low"].shift(3) > df["low"].shift(2)) &
            (df["low"].shift(2) > df["low"].shift(1)) &
            (df["low"].shift(1) > df["low"].shift(0)), "support"] = df["low"]


    df.loc[(df["high"].shift(5) < df["high"].shift(4)) &
    (df["high"].shift(4) < df["high"].shift(3)) &
    (df["high"].shift(3) < df["high"].shift(2)) &
    (df["high"].shift(2) < df["high"].shift(1)) &
    (df["high"].shift(1) < df["high"].shift(0)), "resistance"] = df["high"]

    # Create Simple moving average 30 days
    df["SMA fast"] = df["close"].rolling(30).mean()

    # Create Simple moving average 60 days
    df["SMA slow"] = df["close"].rolling(60).mean()

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=10).rsi()

    # RSI yersteday
    df["rsi yersteday"] = df["rsi"].shift(1)

    # Create the signal
    df["signal"] = 0

    df["smooth resistance"] = df["resistance"].fillna(method="ffill")
    df["smooth support"] = df["support"].fillna(method="ffill")

    
    condition_1_buy = (df["close"].shift(1) < df["smooth resistance"].shift(1)) & \
                        (df["smooth resistance"]*(1+0.5/100) < df["close"])
    condition_2_buy = df["SMA fast"] > df["SMA slow"]

    condition_3_buy = df["rsi"] < df["rsi yersteday"]

    condition_1_sell = (df["close"].shift(1) > df["smooth support"].shift(1)) & \
                        (df["smooth support"]*(1+0.5/100) > df["close"])
    condition_2_sell = df["SMA fast"] < df["SMA slow"]

    condition_3_sell = df["rsi"] > df["rsi yersteday"]

    df.loc[condition_1_buy & condition_2_buy & condition_3_buy, "signal"] = 1
    df.loc[condition_1_sell & condition_2_sell & condition_3_sell, "signal"] = -1


    # Calculamos las ganancias
    df["pct"] = df["close"].pct_change(1)

    df["return"] = np.array([df["pct"].shift(i) for i in range(duration)]).sum(axis=0) * (df["signal"].shift(duration))
    df.loc[df["return"]==-1, "return"] = df["return"]-spread
    df.loc[df["return"]==1, "return"] = df["return"]-spread

    return df#["return"]

def drawdown_function(serie):

    # Calculamos la suma de los rendimientos
    cum = serie.dropna().cumsum() + 1
    
    # Calculamos el máximo de la suma en el período (máximo acumulado) # (1,3,5,3,1) --> (1,3,5,5,5)
    running_max = np.maximum.accumulate(cum)
    
    # Calculamos el drawdown
    drawdown = cum/running_max - 1
    return drawdown


def BackTest(serie, annualiazed_scalar=252):
    # Importar el benchmark
    sp500 = preprocessing_mt5('#US500_Z23', n=7000)['close'].pct_change(1)# cambio porcentual del actual valor con el de 1 atras

    # Cambiar el nombre
    sp500.name = "SP500"

    # Concatenar los retornos y el sp500
    val = pd.concat((serie,sp500), axis=1).dropna()
    # Calcular el drawdown
    drawdown = drawdown_function(serie)*100

    # Calcular el max drawdown
    max_drawdown = -np.min(drawdown)

    # Put a subplots
    fig, (cum, dra) = plt.subplots(1,2, figsize=(20,6))

    # Put a Suptitle
    fig.suptitle("Backtesting", size=20)

    # Returns cumsum chart
    cum.plot(serie.cumsum()*100, color="#39B3C7")

    # SP500 cumsum chart
    cum.plot(val["SP500"].cumsum()*100, color="#B85A0F")

    # Put a legend
    cum.legend(["La prueba", "SP500"])

    # Set individual title
    cum.set_title("Cumulative Return", size=13)

    cum.set_ylabel("Cumulative Return %", size=11)

    # Put the drawdown
    dra.fill_between(drawdown.index,0,drawdown, color="#C73954", alpha=0.65)

    # Set individual title
    dra.set_title("Drawdown de lo que pruebo en SMA_strategy", size=13)

    dra.set_ylabel("drawdown en %", size=11)

    # Plot the graph
    #plt.show()
    st.pyplot(fig)
    # Calcular el índice sortino
    sortino = np.sqrt(annualiazed_scalar) * serie.mean()/serie.loc[serie<0].std()

    # Calcular el índice  beta
    beta = np.cov(val[["return", "SP500"]].values,rowvar=False)[0][1] / np.var(val["SP500"].values)

    # Calcular el índice  alpha
    alpha = annualiazed_scalar * (serie.mean() - beta*serie.mean())

      # Imprimir los estadísticos
    st.write(f"Sortino: {np.round(sortino,3)} ##SI  >1 riesgo menor que rendimiento")
    st.write(f"Beta: {np.round(beta,3)} ##SI  <1 menor b=> menor riesgo")
    st.write(f"Alpha: {np.round(alpha*100,3)} % ##SI  >0 Alpha rendimiento superior al mercado")
    st.write(f"MaxDrawdown: {np.round(max_drawdown,3)} %")