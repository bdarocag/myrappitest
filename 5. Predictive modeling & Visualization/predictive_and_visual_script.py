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
# NOTA 2: Algunas líneas de la configuración estan comentadas por que 
# se ejecutaron múltiples intentos y se verificó que algunos parametros
# afectaban considerablemente el tamaño del error en las predicciones sobre
# el conjunto de prueba.
#
reg = setup(data = data,
target = 'price', 
session_id=1969,
numeric_features=['num_bed', 'num_bath', 'num_floors'], 
categorical_features=['is_waterfront', 'condition','zip'], 
ignore_features=['latitude', 'longitude', 'zip'], 
ordinal_features={'condition':['1','2','3','4','5']}, 
handle_unknown_categorical=True, 
#remove_outliers=True, # AUMENTA RMSE (Parece ser que eliminar los atipicos reduce la varianza y esta es importante para diferenciar los inmuebles más costosos)
#outliers_threshold=0.05, # AUMENTA RMSE
#normalize=True, # La normalización reduce significativamente la varianza y parece ser de gran impacto
#normalize_method= 'minmax', 
#polynomial_features=True, # AUMENTA RMSE Aunque indagando en ejercicios de otras personas, se observa buen resultado, en este caso no fue asi
#trigonometry_features=True, # AUMENTA RMSE """"""                    """"""
#group_features=['schooldist','supermarketdist','warehousedist','churchdist',
#'collegedist','hospitaldist','train_stationdist','universitydist',
#'hangardist'], # Esta estrategia no funcionó como esperaba, lo mejor es utilizar una unica variable ya que todas se correlacionan
#combine_rare_levels=True, # AUMENTA RMSE #Ligera intervención, pero aumenta el error
#rare_level_threshold=0.1, # AUMENTA RMSE
#feature_selection=True, # AUMENTA RMSE #Redundante respecto a la seleccion de variables a partir de la importancia
#feature_selection_threshold=0.6, # AUMENTA RMSE
remove_multicollinearity=True, 
multicollinearity_threshold=0.8, 
#ignore_low_variance=True, 
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
rf = create_model('rf', round = 2)

# %% Extra Trees Regressor
et = create_model('et', round = 2)

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

# %% Random Forest Tuned
rf_tuned = tune_model(rf)

# %% Extra Trees Regressor Tuned
et_tuned = tune_model(et)
# %% Reflexiones a este punto 2
#
# Luego de tunear los parametros de los modelos,
# estos parecen mantenerse estables, pero aumentan ligeramente su error
# en comparación con los modelos con parametros estandar.
#
#
#%% Ver configuracion del mejor modelo
#
# ET TUNED
print(et)
#
# Se observa configuración por si se desea implementar desde la libreria
# de manera directa.
#
# %% Gráfico de residuales
#
plot_model(et, 'residuals') # Residuales

# %% Error
#
plot_model(et, 'error') # Error
#
#
# %% Feature importance
#
plot_model(et, 'feature') # Feature importance
#

# %% Evaluate model
evaluate_model(et)

# %% 
unseen_predictions = predict_model(et, data=data_unseen)
unseen_predictions.head()
# %% Scores
#
# et - R2 = 0.856; Rmse = 137.354 # MEJOR MODELO
# lightgbm - R2= 0.845; Rmse = 142.478
# rf - R2 = 0.844; Rmse = 143.188
# lightgbm_tuned - R2 = 0.825; Rmse = 151.797
# rf_tuned - R2 = 0.820; Rmse = 153.960
# et_tuned - R2 = 0.802; Rmse = 161.192
#
#%% Reflexiones hasta este punto 3
#
# El tuneo de parametros no garantiza una mejora en el error o el desempeño
# del modelo en cuestión. Por el contrario, lo que se hace es testear distintas
# configuraciones completamente diferentes a la estandar. Por lo que de esta forma,
# y para este ejercicio la configuración por defecto obtiene los mejores resultados. 

#%% Save model
#
#
# NOTA IMPORTANTE
# El modelo supera el tamaño limite de git hub, por lo que fue comprimido
# externamente usando el programa 7-zip, para redusir su peso
# en el momento de cargar el modelo debe descomprimirse primero.
save_model(et,'C:/Users/Braya/OneDrive/Documentos/GitHub/myrappitest/5. Predictive modeling & Visualization/Final et Model 1Dec2022')
# %%
