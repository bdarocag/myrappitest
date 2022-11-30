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

#%% Lectura de datos iniciales
house_sales = pd.read_csv('C:/Users/Braya/Documents/GitHub/myrappitest/Data/house_sales.csv', sep =',')

# %%
