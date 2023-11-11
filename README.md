# <h1 align=center> **Deteccion de Emociones** </h1>


<p align="center">
<img src="https://github.com/rodcordova/Deteccion-de-emociones/blob/master/images/portada.png"  height=300>
</p>

## Proyecto de Análisis de Sentimientos de Criptomonedas

## Introducción

Este proyecto tiene como objetivo realizar el análisis de sentimientos de una criptomoneda seleccionada utilizando diversas fuentes de información y técnicas de Recuperación de Información. La criptomoneda elegida es [Bitcoin] ($[BTC]).

## Configuración del Proyecto
El proyecto se divide en 3 notebooks en las cuales se analizan datos extraidos de la web ya sea por apis o por webscraping y un archivo funciones.py en la cual tendra funciones que sean comunes en las notebook

### Base de Datos

Se utiliza [mysql] como base de datos para almacenar la información analizada. Puedes encontrar el esquema de la base de datos en el archivo [criptomoneda.sql].


### Obtención de Datos

Se accede a diversas fuentes de información, incluyendo:

- Newsapi
- Bing
- Reddit
- CoinTelegraph


## Instrucciones de Ejecución

1. Clona este repositorio: `git clone https://github.com/rodcordova/Deteccion-de-emociones`
2. Instala las dependencias: `pip install -r requirements.txt`
3. Ejecuta el código principal: `Los notebooks`
4. Ejecutar una interfaz para uso interactivo: `https://deteccion-de-emociones-olmpcph6mez8bqone98q4o.streamlit.app/`

