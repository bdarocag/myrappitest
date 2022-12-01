#%% Information
#
# Date 30/11/2022
# Brayan David Aroca Gonzalez
# Data Scientist Senior
# MsC. Biostatistics
#
# La idea principal de este script es recopilar información adicional a los datos
# entregados. Si bien, estos pueden ser muy completos, conozco algunas formas
# de enriquecer los datos y averiguar si nuevas variables influyen en los precios
# de las viviendas en Seattle
#
#
#%% Modules
import os # Directorios y archivos
import pandas as pd # Data Frames
import numpy as np # Arrays
import matplotlib as plt # Plots
import seaborn as sns # Graphs
import osmnx as ox # Open Street Maps module
import geopandas as gpd
from geopy.distance import geodesic # Calculate distances

#%% Lectura de datos iniciales
house_sales = pd.read_csv('C:/Users/Braya/OneDrive/Documentos/GitHub/myrappites'\
    't/Data/house_sales.csv', sep =',')

# %% Descripción de variables
house_sales.describe()

# Podemos ver variable por variable, una descripcion inical con diferentes me-
# didas de tendencia central y/o su distribución respecto a estas.
#
# %% Mineria de información espacial
#
# Como este es el apartado de mineria de datos, lo hraé respecto a la variable
# que considero es más importante para este proposito en este dataset, las
# coordenadas.
# 
# Utilizaré estas para calcular la distancia respecto a hospitales, universidades,
# colegios, estaciones de transporte, etc; esto con el proposito de incrementar la
# información relacionada al precio de una vivienda.
#
place_name = 'Seattle' # Sitio de busqueda
area = ox.geocode_to_gdf(place_name) #Obtener poligono del sitio
type(area) # Verificar tipo de dato
area.plot() # Verificar poligono en grafico

# %% Obtener escuelas, universidades, etc
#
etiquetas = ['school', 'supermarket', 'warehouse', 'church', 'college',
'hospital', 'train_station', 'university', 'hangar']

#%% Loop para obtener nuevas features
#
# NOTA: Este segmento se demora en su ejecución,
# en mi equipo se tomó 36 minutos.
#
# Ryzen 5 3600
# RTX 3060Ti
# SSD 1TB
#
#
# Los datos seran almacenados para cargarlos directamente. Ver siguiente sección.


for x in range(0, len(etiquetas)):
    etiqueta = {'building':etiquetas[x]}
    poligonos = ox.geometries_from_place(place_name, etiqueta)
    
    # Extraer todos los poligonos de escuelas
    poligonos  = poligonos.loc[:,poligonos.columns.str.contains('addr:|geometry')]
    
    #Extraer coordenadas
    poligonos = poligonos.to_crs(4326)
    poligonos['Longitud'] = poligonos.centroid.x # Long
    poligonos['Latitud'] = poligonos.centroid.y # Lat
    
    # Crear lista vacia para almacenar distancia minima
    distancia_final = []
    for i in range(0, len(house_sales)): #Correr loop sobre cada una de las viviendas
        distances = [] # Crear lista vacia para almacenar distancias
        
        for j in range(0, len(poligonos)): #Correr loop sobre cada poligono
            distance_in_km = geodesic((house_sales['latitude'][i],house_sales['longitude'][i]), 
            (poligonos['Latitud'][j], poligonos['Longitud'][j])).km #Calcular distancia en Km
            distances.append(distance_in_km) #Insertar distancia calculada en distancias
        distancia_final.append(np.min(distances)) #Obtener distancia minima
    house_sales[etiquetas[x] + 'dist'] = distancia_final # Crear columna con datos
    house_sales[etiquetas[x] + 'dist'] = house_sales[etiquetas[x] + 'dist'].round(decimals = 2) # Olvide redondear en el loop :')

#%% Guardar nuevo archivo con nuevas features

pd.to_pickle(house_sales, 'C:/Users/Braya/OneDrive/Documentos/GitHub/myrappit'\
    'est/Data/house_sales_ext.pickle')
# %% Final del ejercicio
#
# Al final, se obtiene una columna nueva con la distancia a la infrestructura
# más cercana, por cada tipo de infraestructura.
# Estas nuevas variables las estudiare en el siguiente paso.
