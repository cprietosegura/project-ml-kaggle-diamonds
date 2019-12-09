from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.cleaning import preparingDataset, preparingDatasetBis
from src.training import getModelScore, saveTrainedModel
from src.submission import prepareSubmission

diamonds=preparingDataset('./input/data.csv')

models = {
    'Linear Regression': LinearRegression(),
    "Lasso Regression": linear_model.Lasso(alpha=0.1),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=250),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=0),
}

models_2submission={
    'LassoLars': linear_model.LassoLars(),
    'SVR': SVR()
}
#inicialmente utilizo este bucle para probar varios modelos, después me quedo con Random Forest Regressor
#for modelName, model in models.items():
#    print("Training model: {}".format(modelName))
#    mod=getModelScore(diamonds, model)
#    saveTrainedModel(mod,"{}".format(modelName))
#    prepareSubmission('./models/{}.sav'.format(modelName),'{}'.format(modelName))
#    print('CSV Generado')


print('Entrenando modelo')
randomf=(getModelScore(diamonds,RandomForestRegressor(n_estimators=250)))

print('Modelo entrenado')
saveTrainedModel(randomf,'randomforest')

print('Modelo salvado')
prepareSubmission('./models/randomforest.sav','randomforest')

print('CSV generado')