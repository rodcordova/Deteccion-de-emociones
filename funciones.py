import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cycler

import nltk
from wordcloud import WordCloud #
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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