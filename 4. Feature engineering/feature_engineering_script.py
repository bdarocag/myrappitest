#%% Ingenieria de datos
#
# A partir de la exploración pudimos darnos cuenta de algunas variables con
# especiales caracteristicas respecto a nuestra problematica como lo son el
# número de baños, tamaño de la vivienda, entre otras.

# Adicionalmente, vimos que por defecto algunas otras no son reconocidas según
# su clase (numerica, categorica / discreta, continua, ordinal, etc).

# Las variables obtenidas a partir de la mineria, entre sí se correlacionan
# lo que puede acarrear problemas de multicolinealidad. Para esto ajustaremos un threshold
# o umbral de aceptacion en el cual solo se permitiran aquellas con menos de
# 85% de inter-correlación. Asi mismo se aplicará a las demás variables.

# Otro de los objetivos de la ingenieria de datos es posiblemente, constituir relaciones
# entre las variables para explicar la variable dependiente a partir de relaciones
# no lineales, estas pueden ser polinomiales o trigonometricas, para esto configuraremos entre
# las variables numericas este tipo de relaciones y veremos su implicación en el modelo

# Encoding de variable categorias.
# Estrategia aplicada para estudiar mejor los niveles de las categorias y su
# efecto en la variable dependiente.

# Finalmente, se aplicara una normalización con el fin de reducir el rango de interpretacion
# sin perdida de información.

#%% Modulos
from pycaret.regression import * # Py Caret - Framework para ML
import os # Directorios y archivos
import pandas as pd # Data Frames
import numpy as np # Arrays
import pickle #Save and Load objects
#%% Cargar datos
with open(r"C:/Users/Braya/OneDrive/Documentos/GitHub/myrappitest/Data/house_sales_ext.pickle", "rb") as input_file:
    house_sales = pickle.load(input_file) 

#%% Inicialización del modelado

reg = setup(data = house_sales, #Conjunto de datos
target = 'price', # Variable dependiente ('Target')
session_id=1969, # Semilla
categorical_features=['is_waterfront', 'condition','zip'], #Especificar variables categoricas
ignore_features=['latitude', 'longitude'], # Variables a omitir
ordinal_features={'condition':['1','2','3','4','5']}, # Encoding ordinal
handle_unknown_categorical=True, # Manejar categorias desconocidas
remove_outliers=True, # Remover datos atipicos
outliers_threshold=0.05, #Porcentaje máximo de datos removidos siendo outliers
normalize=True, # Normalizacion de numericas
normalize_method= 'minmax', # Normalizacion en rango 0 a 1
polynomial_features=True, # generar relaciones polinomiales
trigonometry_features=True, #generar relaciones trigonometricas
group_features=['schooldist','supermarketdist','warehousedist','churchdist',
'collegedist','hospitaldist','train_stationdist','universitydist',
'hangardist'], #Agrupar variables para obtener un valor promedio maximo mediano y ver como se comporta
combine_rare_levels=True, #Combinar en una sola categira, las categorias más dispersas (variables categoricas)
rare_level_threshold=0.1, # Umbral para combinar
feature_selection=True, # Seleccion de variables a partir de modelos internos
feature_selection_threshold=0.6, #umbral para mantener cierta cantidad de variables
remove_multicollinearity=True, # Remover multicolinealidad
multicollinearity_threshold=0.7, # Umbral para remover la multicolinealidad
ignore_low_variance=True, # Descartar variables con mala distribucion y baja varianza
experiment_name='regresion', # Nombre para identificar intento
fold_strategy='stratifiedkfold', # Metodo de validacion cruzada
fold=10, # Cantidad de divisiones en la data
use_gpu=True) # utilizar GPU cuando sea posible
# %%
#
# De esta forma quedan configurados todos los parametros para la ejecución
# de los modelos en el siguiente paso '5. Predictive modeling...'