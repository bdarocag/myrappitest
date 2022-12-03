# Test to apply a Rappi's job oportunity

<div>
  <p align="center">
    <img src="Images/logo-rappi.svg" width="800"> 
  </p>
</div>

# Descripción

Este ejercicio corresponde al desarrollo y aplicación de un modelo de Machine Learning para la predicción de precios de inmuebles en Seattle, este ejercicio, surge como una prueba para aplicar al rol de Data Scientist en Rappi. A partir del archivo `house_sales.csv` se debe aplicar conocimientos en **Ciencia de datos** para lograr el objetivo de predecir el valor de las casas a partir de la información brindada y/o aquella que se pueda minar adicionalmente. Para realizar este trabajo, se utilizó el framework de Machine Learning [PyCaret](https://github.com/pycaret/pycaret).

[![made-with-python](https://img.shields.io/badge/Made%20with-Python%203.8-1f425f.svg?logo=python)](https://www.python.org/)

## Table of contents

* [Descripción](#Descripción)
* [Instrucciones](#Instrucciones)
  + [Nota](#Nota)
* [0. Entendimiento del problema](#0. Entendimiento del problema)
  + [Noticias locales](#Noticias locales)
* [1. Data mining](#1. Data mining)
* [2. Data cleaning](#2. Data cleaning)
* [3. Data exploration](#3. Data exploration)
  + [EDA Report](#EDA Report)
* [4. Feature engineering](#4. Feature engineering)
* [5. Predictive modeling](#5. Predictive modeling)
  + [Saved model](#Saved model)
* [6. Load and use model](#6. Load and use model)
* [Conclusiones](#academic-publications)


# Instrucciones

La forma de abordar este ejercicio es siguiendo el clásico ciclo de vida en ciencia de datos, para mayor comodidad, las carpetas han sido numeradas en orden (paso a paso). Asi mismo, cada paso cuenta con un script en donde este se ha desarrollado. Los scripts cuentan con separadores de linea `#%%` en su mayoria titulados para indicar la accion correspondiente. Es importante leer adecuadamente los comentarios dentro de los scripts pues contienen las distintas apreciaciones y explicaciones que consideré necesarias.

# Nota
Es importante tener en cuenta que en los pasos **1. Data mining**, **5. Predictive modeling** y **6. Load and use model**, hay instrucciones especiales en cada uno de los scripts correspondientes. Por esto motivo se recuerda leer con especial atención dichas indicaciones.


# 0. Entendimiento del problema

En este espacio, se encontrara un achivo *.md* con una pequeña profundización realizada sobre el problema de este ejecrcicio. Allí se indica como se abordo la problematica y como se pretendio entender.

# Noticias locales
Dentro de la carpeta se indica un reporte sobre los precios de inmuebles en Seattle, donde se indican tendencias y expectativas del mercado inmobiliario para esta ciudad.


# 1. Data mining
Esta sección aplica estrategias de mineria de datos para obtener información complementaria y enriquecer los datos originales.


# 2. Data cleaning
Como los datos se encontraban en muy buen estado y no presentaban valores nulos o vacios, esta sección es diminuta y únicamente de consulta.
Como dato extra, algunas estrategias de manejo de datos son aplicadas en la sección [4. Feature engineering](#4. Feature engineering).


# 3. Data exploration
Clásico proceso de exploración de datos, en donde se consultan las medidas de tendencia central de las variables, se verifica su distribución, se hacen comparaciones sencillas, se indaga sobre niveles de correlación y con ello se plantean hipotesis de trabajo.

# 4. Feature engineering
Acá se aplica todo el poder del framework [PyCaret](https://github.com/pycaret/pycaret) para ejecutar tareas simples y complejas de ingeniería de datos. Aquí se aplica normalización, omisión de variables segun indicadores de multicolinealdiad, transformaciones polinomiales y trigonometricas, agrupacion de variables entre otras. Como este framework permite una aplicación de estas técnicas de manera sencilla, solo se instruye la configuración utilizada.


# 5. Predictive modeling
Utilizando la configuración del paso [4. Feature engineering](#4. Feature engineering), se inicializa el proceso de modelado una y otra vez. En este segmento hallaran el script final, el cuál resultó en el mejor modelo obtenido. Esto quiere decir que las configuraciones del paso [4. Feature engineering](#4. Feature engineering), no fueron utilizadas en su totalidad. Por tal motivo encontraran varias lineas comentadas y una pequeña justificación.

# Saved model
Durante este paso, se almacena el modelo en un archivo ZIP por que su peso excede los limites de GitHub para su almacenamiento. Por tal motivo, este archivo debe descargarse y descomprimirse antes de ser utilizado en el paso [6. Load and use model](#6. Load and use model)


# 6. Load and use model
En el paso final, se indica mediante un script comentado, los pasos a seguir para cargar el modelo elaborado y aplicarlo a un cojunto de datos nuevo. Es preciso en este apartado leer con atención los comentario realizados.


# Conclusiones
- El modelo resultante obtuvo valores en sus metricas bastante positivos, que a su vez se mantuvieron en la implementación de un conjunto de datos "nuevo" (en realidad era el mismo). De acuerdo a estos valores, se espera un desempeño optimo cuando se aplique sobre datos verdaderamente nuevos.
- Es posible mejorar el modelo mediante técnicas de ensamble o conglomeración, así mismo podria evaluarse el uso de técnicas con redes neuronales.
- A consideración personal, los métodos basados en regresión, son lo suficientemente optimos para predecir la variable objetivo dentro de este ejercicio.
- Se debe ser cuidadosos a la hora de aplicar técnicas de normalización o ingenieria de datos, mcuhas veces las transformaciones re escalan los datos haciendo que se pierda variabilidad y con ello propiedades que diferencian los registros unos de otros.

# !Gracias
