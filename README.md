# Project ML | Diamonds kaggle competition

En este proyecto he desarrollado un modelo de machine learning con la librería `Scikit-learn` de `Python` para predecir el precio de los diamantes a partir de un dataset de Kaggle. Asimismo, el proyecto forma parte de una competición de Kaggle.

## Proceso

### Limpieza y selección de atributos
El primer paso fue analizar la información del dataset con `Pandas` y evaluar cuáles eran las mejores features para entrenar el modelo de machine learning. Para ello, transformé las variables categóricas en numéricas y posteriormente elaboré una matriz de correlación y un mapa de calor con `seaborn` con el objetivo de descubrir las variables con alto nivel de correlación.  

### Entrenamiento de modelos
Tras la fase de limpieza y selección de atributos, comencé a entrenar varios modelos. Rápidamente comprobé que debía usar modelos de regresión y no de clasificación. Así, entrené varios modelos del primer tipo: Linear Regression, Lasso Regression, Gradient Boosting Regressor, KNeighborsRegressor, Random Forest Regressor y DecisionTreeRegressor. 

Después de entrenar y evaluar los diferentes resultados a través del Mean Absolute Error, Mean Squared Error y el R2 square value, llegué a la conclusión de que el mejor modelo era el de `Random Forest Regressor` con un R2 square value entorno al 98% de acierto.

Finalmente utilicé la librería `Pickle` para salvar el modelo entrenado y aplicarlo en el dataset test para participar en la competición de Kaggle.

### Overview

![alt text](https://github.com/cprietosegura/project-ml-kaggle-diamonds/blob/master/output/overview.jpg)

