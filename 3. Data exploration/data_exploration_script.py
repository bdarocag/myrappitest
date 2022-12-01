#%% Análisis Exploratorio de Datos - Script
#
#
# Para continuar, luego de  hacer una verificación sobre la existencia de datos
# vacios o nulos. Y como la data se encontraba impecable en ese sentido, podemos
# empezar a observar y analizar el comportamiento de los datos, respecto de la
# variable objetivo, el precio de venta.
#
#
#%% Modulos
import os # Directorios y archivos
import pandas as pd # Data Frames
import numpy as np # Arrays
import pickle #Save and Load objects
from pandas_profiling import ProfileReport # EDA html report

#%% Cargar datos
with open(r"C:/Users/Braya/OneDrive/Documentos/GitHub/myrappitest/Data/house_sales_ext.pickle", "rb") as input_file:
    house_sales = pickle.load(input_file) 
# %% Exploration
#
# El objetivo de esta sección es formar hipotesis sobre nuestro problema,
# en este caso, se considera que el precio de los inmuebles puede explicarse
# a partir de las variables que acompañan este conjunto de datos, son variables
# con informacion directa del inmueble en cuestión, o de caracteristicas del
# lugar donde este se encuentra.

#%% Describir todas las variables
house_sales.describe()
# %% Aplicar EDA automatizado
#
profile = ProfileReport(house_sales)
profile.to_file(output_file='C:/Users/Braya/OneDrive/Documentos/GitHub/myrapp'\
    'itest/3. Data exploration/house_sales_report.html')

# %% Reflexiones
#
# El reporte es muy completo y puede observarse al abrirse en cualquier navegador
# o compilador de Html.
#
# En primer lugar al ir directamente a la variable objetivo, vemos que esta tiene
# fuertes relaciones con otras tres variables.
# 
# Del resultado de extraer las distancias a las diferentes infraestructuras,
# se concluye que entre estas hay una alta correlación y repecto al precio
# la correlación es negativa en todos los casos (entre mas distancia hay menor
# es el precio del inmueble, algo lógico).
#
# El reporte no logra distinguir algunas variables categoricas como el código ZIP
# una categorica ordinal como el año de remodelación, que a mi concepto se encuentra
# muy desbalanceada para contemplarse dentro del modelaje.
#
# Gracias a esta exploración se puede proceder a evaluar sobre un modelos, bien configurados,
# las relaciones de los features sobre la variable de respuesta 'price', con una idea clara
# sobre que variables posiblemente si sean importantes y cuales no.
