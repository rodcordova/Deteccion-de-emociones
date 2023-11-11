import streamlit as st
from sqlalchemy import create_engine,text #se lleva bien 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cycler
import joblib
#from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer

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

#################################
st.title('Series temporales')
st.markdown('***')



