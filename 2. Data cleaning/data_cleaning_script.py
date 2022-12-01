#%% Limpieza de datos - Script
#
#
# Para continuar, luego de agregar más caracteristicas/variables a nuestros
# datos, es pertinente hacer una verificación sobre la existencia de datos
# vacios, nulos.
#
#
#%% Modulos
import os # Directorios y archivos
import pandas as pd # Data Frames
import numpy as np # Arrays
import pickle #Save and Load objects

#%% Cargar datos
with open(r"C:/Users/Braya/OneDrive/Documentos/GitHub/myrappitest/Data/house_sales_ext.pickle", "rb") as input_file:
    house_sales = pickle.load(input_file) 
# %% Check Nan or Null Values
print(house_sales.isnull().values.any())
print(house_sales.isnull().sum().sum()) # Ningun valor nulo o vacio
