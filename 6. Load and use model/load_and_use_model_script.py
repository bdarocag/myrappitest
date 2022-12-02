#%% Cargar el modelo y utilizarlo
#
# En primer lugar descomprimiremos el ZIP ubicado en la carpeta 5. Predictive...
# allí debio queda el modelo llamado 'Final et Model 1Dec2022.pkl'
# podemos ejecutar el resto de este código y remplazar los directorios segun
# sea el caso.
#
# Los posibles cambios serán indicado usando '<-----'
# con una breve explicación.

#%% Modulos
#
# Modulos necesarios para ejecutar el modelo
from pycaret.regression import * # Py Caret - Framework para ML
from pycaret.utils import check_metric # Chequear las metricas
import os # Directorios y archivos
import pandas as pd # Data Frames
import numpy as np # Arrays
import pickle #Save and Load objects
#%% Cargar datos
#
# <------- En este punto se deben cargar los datos nuevos,
# <------- o los datos reales de prueba del modelo
# <------- Si es asi, modificar la ruta y el archivo.
with open(r"C:/Users/Braya/OneDrive/Documentos/GitHub/myrappitest/Data/house_sales_ext.pickle", "rb") as input_file:
    new_data_house_sales = pickle.load(input_file)

#%% Cargar el modelo
#
# En primer lugar se debe descargar el archivo Zip
# Final et Model 1Dec2022.zip
# y descomprimirlo en el directorio deseado
# Luego se procede a cargar el modelo de la siguiente forma
#
#
# <------- Cambiar el directorio al sitio donde descargo el modelo
# <-------
# <-------
modelo = load_model('C:/Users/Braya/Downloads/Final et Model 1Dec2022')
# %% Aplicar modelo sobre data nueva
#
#
new_prediction = predict_model(modelo, data=new_data_house_sales)

# %% Ver resultados de la prediccion
#
# Al final de los datos (la ultima columna), se agregara una nueva llamada 'Label'
# con las predicciones.
new_prediction.head()
# %% Chequear la metrica resultante
check_metric(new_prediction.price, new_prediction.Label, 'R2')
# %%
check_metric(new_prediction.price, new_prediction.Label, 'RMSE')
# %%
# Sobre los mismos datos que se utilizaron para elaborar el modelo
# los resultados se ven prometedores y más optimos (Obviamente)
#