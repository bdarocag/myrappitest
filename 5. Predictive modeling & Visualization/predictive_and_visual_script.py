#%% Predicciones y visualizacion de resultados
#
# En este momento realizaremos la implementación de múltiples algorithmos de
# regresión para explicar el precio de los inmuebles a partir de las variables
# que como hipotesis los caracterizan.
#
# Posterior a la ejecución se realizarán distintas visualizaciones para 
# evaluar el comportamiento de las variables y del modelo
#
# Aplicaremos un tuneo de los parametros con el fin de mejorar el ajuste y las
# metricas de error
#%% Modulos
from pycaret.regression import * # Py Caret - Framework para ML
import os # Directorios y archivos
import pandas as pd # Data Frames
import numpy as np # Arrays
import pickle #Save and Load objects
#%% Cargar datos
with open(r"C:/Users/Braya/OneDrive/Documentos/GitHub/myrappitest/Data/house_sales_ext.pickle", "rb") as input_file:
    house_sales = pickle.load(input_file) 

# %% Separacion conjuntos de datos
data = house_sales.sample(frac=0.9, random_state=1969)
data_unseen = house_sales.drop(data.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions ' + str(data_unseen.shape))

#%% Inicialización del modelado
#
# NOTA: La configuración del paso anterior
reg = setup(data = data,
target = 'price', 
session_id=1969,
numeric_features=['num_bed', 'num_bath', 'num_floors'], 
categorical_features=['is_waterfront', 'condition','zip'], 
ignore_features=['latitude', 'longitude'], 
ordinal_features={'condition':['1','2','3','4','5']}, 
handle_unknown_categorical=True, 
#remove_outliers=True, # AUMENTA RMSE
#outliers_threshold=0.05, # AUMENTA RMSE
normalize=True, 
normalize_method= 'minmax', 
#polynomial_features=True, # AUMENTA RMSE
#trigonometry_features=True, # AUMENTA RMSE
group_features=['schooldist','supermarketdist','warehousedist','churchdist',
'collegedist','hospitaldist','train_stationdist','universitydist',
'hangardist'], 
#combine_rare_levels=True, # AUMENTA RMSE
#rare_level_threshold=0.1, # AUMENTA RMSE
#feature_selection=True, # AUMENTA RMSE
#feature_selection_threshold=0.6, # AUMENTA RMSE
remove_multicollinearity=True, 
multicollinearity_threshold=0.7, 
ignore_low_variance=True, 
experiment_name='regresion', 
fold_strategy='stratifiedkfold', 
fold=10, 
use_gpu=True)
#
#
# Si parece no avanzar, no olvidar hacer click en los simbolos </> </> </>
# 
# %% Ejecutar todos los modelos de regresion
modelos = compare_models() # Ejecución
# Si parece no avanzar, no olvidar hacer click en los simbolos </> </> </>
#
# De acuerdo al resultado [Una vez Termine], los mejores modelos de acuerdo a 
# las distintas metricas de error, son:
# Lightgbm - Light Gradient Boosting Machine
# Xgboost - Extreme Gradient Boosting
#
# Realizaremos el tuning sobre estos y los aplicaremos sobre el conjunto de
# prueba para verificar sus resultados
#
#
# %% Modelos individuales
#
#%% Lightgbm
lightgbm = create_model('lightgbm', round = 2)
# %% Xgboost
xgboost = create_model('xgboost', round = 2)

# %% Reflexiones a este punto
#
# Después de aplicar modelos individuales con validacion cruzada en el training set
# y 10 k-folds, lightgbm se mantiene ligeramente mejor que xgboost
#
#
#%% Tunear modelos
#
#%% Lightgbm Tuned
lightgbm_tuned = tune_model(lightgbm)

# %% Xgboost Tuned
xgboost_tuned = tune_model(xgboost)
# %% Reflexiones a este punto 2
#
# Luego de tunear los parametros de los modelos,
# Xgboost, obtiene una reduccion importante en el error.
# Por este motivo, elegiré este modelo como el mejor y lo explorare para
# desentrañar su resultado aplicandolo en el conjunto de prueba.
#
#
#%% Ver configuracion del mejor modelo
#
# XGBOOST TUNED
print(xgboost_tuned)
#
# Se observa configuración por si se desea implementar desde la libreria
# de manera directa.
#
# %% Gráfico de residuales
#
plot_model(xgboost_tuned, 'residuals') # Residuales

# %% Error
#
plot_model(xgboost_tuned, 'error') # Error
#
# No me gustó tanto la distribución del error pero prefiero esperar
# al aplicarse en el verdadero conjunto de prueba.
# %% Feature importance
#
plot_model(xgboost_tuned, 'feature') # Feature importance
#

# %% Evaluate model
evaluate_model(xgboost_tuned)

# %%
unseen_predictions = predict_model(xgboost_tuned, data=data_unseen)
unseen_predictions.head()
# %%
