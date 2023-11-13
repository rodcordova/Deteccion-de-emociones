import streamlit as st
from sqlalchemy import create_engine,text #se lleva bien 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cycler
import joblib
#from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from funciones import support_resistance,preprocessing_mt5,drawdown_function,BackTest

## series
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric

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

## HISTOGRAMA
# Cargamos datos desde mysql

my_con=create_engine('mysql+pymysql://root:root1234@localhost:3306/nlp')

query = text('''
    SELECT * FROM frec_bing_btc   
''')
df_read = pd.read_sql_query(query, my_con)

query_news = text('''
    SELECT * FROM frec_newsapi_btc   
''')
df_newsapi = pd.read_sql_query(query_news, my_con)

# mostrar dataframe
#if st.checkbox('mostrar dataframe'):
#    st.dataframe(df_read)
st.title('Reconocimiento de emociones')

st.markdown('***')
# figura frecuencia
limite=st.slider('Histograma de palabras mas frecuentes',0,100,20)
col1 , col2 = st.columns(2)
with col1:
    fig=plt.figure(figsize=(15,8))
    plt.title('Gráfico de Frecuencia (Bing)')
    plot=sns.barplot(x=df_read.iloc[:limite].Word,y=df_read.iloc[:limite].Frecuencia)
    for item in plot.get_xticklabels():
        item.set_rotation(45)
    st.pyplot(fig)

with col2:
    fig=plt.figure(figsize=(15,8))
    plt.title('Gráfico de Frecuencia (Newsapi)')
    plot=sns.barplot(x=df_newsapi.iloc[:limite].Word,y=df_newsapi.iloc[:limite].Frecuencia)
    for item in plot.get_xticklabels():
        item.set_rotation(45)
    st.pyplot(fig)

## SENTIMIENTO
query_bing_sen = text('''
    SELECT * FROM bing_btc   
''')
df_read_sen = pd.read_sql_query(query_bing_sen, my_con)
bing_valor=sum([val for val in df_read_sen['sentiment_analysis']])/df_read_sen.shape[0]

query_news_sen = text('''
    SELECT * FROM newsapi_btc   
''')
df_newsapi_sen = pd.read_sql_query(query_news_sen, my_con)
news_valor=sum([val for val in df_newsapi_sen['sentiment_analysis']])/df_newsapi_sen.shape[0]




# QUITARLO
# Crear un DataFrame vacío
df = pd.DataFrame(columns=['Texto'])


# Recopilar texto desde el usuario
nuevo_texto = st.text_area('Ingrese noticia que quiera analizar y presione Enter:')
boton=st.button('Guardar Texto')
# Verificar si se ingresó texto y si se presionó Enter
if nuevo_texto and boton:
    # Agregar el nuevo texto al DataFrame
    #df = df.append({'Texto': nuevo_texto}, ignore_index=True)
    st.success('¡Texto guardado con éxito!')
# Cargamos el modelo
best_model=joblib.load('best_model_sentiment.plk')
cou_vec=joblib.load('count_vectorizer.pkl')

# Realizar la predicción
prediccion=best_model.predict(cou_vec.transform([nuevo_texto]).toarray())[0]
#prediccion = best_model.predict(nuevo_texto.toarray())

st.write("0: negativa,1: positiva,2: neutra")
if st.checkbox('Mostrar sentimiento'):
    st.write('Bing',bing_valor)
    st.write('News',news_valor)
    st.write('Manual',prediccion)

################################# CARTERA
st.title('Diversificar cartera')

st.markdown('***')
########
# Área de entrada de texto
lista = st.text_input("Ingresa tu lista separada por comas",'BITCOIN')
# Convertir el texto ingresado en una lista
listnames = [item.strip() for item in lista.split(',')]
returns = pd.DataFrame()
boton=st.button('Guardar activos')
# Verificar si se ingresó texto y si se presionó Enter
if listnames and boton:
    # Agregar el nuevo texto al DataFrame
    #df = df.append({'Texto': nuevo_texto}, ignore_index=True)
    st.success('¡Texto guardado con éxito!')

# Calcular la rentabilidad de cada estrategia
for name in listnames:

    returns[name] = support_resistance(name,mt5_live=True,n=7000, duration=10, spread=0.01)['return']

# Representar los resultados
cumulative_returns =returns.fillna(value=0).cumsum()*100
# Crear un gráfico con matplotlib
fig, ax = plt.subplots(figsize=(15, 6))
cumulative_returns.plot(ax=ax)

ax.set_title('Rendimientos Acumulativos')
ax.set_xlabel('Período')
ax.set_ylabel('Rendimiento Acumulativo (%)')
st.pyplot(fig)

#####
returns["portfolio"] = returns.sum(axis=1)/returns.shape[1]
# Dataframe vacío
values = pd.DataFrame(index=["RETURN", "DRAWDOWN", "RETURN DRAWDOWN RATIO"])
# Calculamos retorno/drawdown
for col in returns.columns:
    ret = (returns[col].dropna().cumsum().iloc[-1])
    dd = -np.min(drawdown_function(returns[col].dropna()))
    ret_dd = ret/dd
    
    values[col] = ret,dd,ret_dd

cartera=values.transpose().sort_values(by="RETURN DRAWDOWN RATIO", ascending=False)
# mostrar dataframe
#if st.checkbox('mostrar dataframe'):
st.dataframe(cartera)

########
# Área de entrada de texto
cripto = st.text_input("Ingresa la criptomoneda a analizar con la estrategia", "BITCOIN")
boton=st.button('Guardar cripto')
# Verificar si se ingresó texto y si se presionó Enter
if cripto and boton:
    # Agregar el nuevo texto al DataFrame
    #df = df.append({'Texto': nuevo_texto}, ignore_index=True)
    st.success('¡Texto guardado con éxito!')

resul=support_resistance(cripto)
#st.hist(resul['return'])
# Crear un histograma con Matplotlib
fig, ax = plt.subplots(figsize=(15, 6))
ax.hist(resul['return'], bins=30, color='pink', alpha=0.7)

ax.set_title('Histograma de Retornos')
ax.set_xlabel('Retornos')
ax.set_ylabel('Frecuencia')
st.pyplot(fig)

strategy=support_resistance("BITCOIN",n=7000, mt5_live=True) # para año mes dia (con api) ## cartera PYPL, PFE (farmaceutica)
dfc = strategy['return'].loc["2020":]-0.00001 # coste de transaccion .00001

BackTest(dfc, 252)


fig, ax = plt.subplots(figsize=(15, 6))
strategy[["close", "SMA fast", "SMA slow"]].loc["2023-04-12 22:00:00":].plot(ax=ax)

ax.set_title('Rendimientos Acumulativos')
ax.set_xlabel('Período')
ax.set_ylabel('Rendimiento Acumulativo (%)')
st.pyplot(fig)

############### SERIES TEMPORALES
st.title('Series Temporales')

st.markdown('***')


periodo_s = st.text_input(f"Ingresa peiodo en hs del {cripto}", "24")
boton_s=st.button('Guardar periodo')
# Verificar si se ingresó texto y si se presionó Enter
if periodo_s and boton_s:
    # Agregar el nuevo texto al DataFrame
    #df = df.append({'Texto': nuevo_texto}, ignore_index=True)
    st.success('¡Texto guardado con éxitos!')

# Cargamos el modelo
model_serie=joblib.load('serie_temporal.pkl')

future = model_serie.make_future_dataframe(periods=int(periodo_s), freq='H') #32 semanas tiene un año
forecast = model_serie.predict(future)

#model_serie.plot(forecast,xlabel='Fecha',ylabel='Fecha')
#plt.title('Bitcoin',fontsize=13)
#plt.show()

fig, ax = plt.subplots(figsize=(15, 6))
model_serie.plot(forecast,ax=ax)

ax.set_title(f'{cripto}')
ax.set_xlabel('Fecha')
ax.set_ylabel('Precio')
st.pyplot(fig)

#df_cv=cross_validation(model_serie, horizon=periodo_s+" hours", initial="120 hours")
#fig = plot_cross_validation_metric(df_cv, metric='rmse')
#st.pyplot(fig)
#df_p = performance_metrics(df_cv)
#st.write('El RMSE es:', df_p.rmse.mean())

#st.write(f'Precio para la fecha {} del {} es: ', )

fecha = st.text_input(f"Ingresa fecha para predecir el valor de {cripto} ", "2023-11-12 04:00:00")
boton_f=st.button('Guardar fecha')
# Verificar si se ingresó texto y si se presionó Enter
if fecha and boton_f:
    # Agregar el nuevo texto al DataFrame
    #df = df.append({'Texto': nuevo_texto}, ignore_index=True)
    st.success('¡Texto guardado con éxitos!')

# 1c) predecimos etiqueta de uba salida
future_date = pd.to_datetime(fecha)
future = pd.DataFrame({'ds':[future_date]})
# predecimos
forecast = model_serie.predict(future)
predicted_price=forecast['yhat'].iloc[0]
st.write(f'El precio predicho por serie temporal para {cripto} es de: ', predicted_price)

######################## Prediccion
best_pipe=joblib.load('model_best_pipeline.plk')

fecha_numeros=(pd.to_datetime(fecha) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

input_data = {
        #"open": [openai],
        #"high": [high],
        #"low": [low],
        #"volume": [volume],  # Agregar el nombre del nuevo juego
        "fecha_numeros":[fecha_numeros]
}
input_df = pd.DataFrame(input_data)

# Convertir el diccionario en una matriz bidimensional
input_array = np.array(list(input_data.values())).reshape(1, -1)

# Luego, puedes usar input_array para hacer la predicción
predicted_price = best_pipe.predict(input_array)

st.write(f'El precio predicho por Prediccion ML para {cripto} es de: ', predicted_price[0])