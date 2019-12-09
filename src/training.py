from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pickle

def getModelScore(df, model_function):
    X=df[['carat','depth', 'table', 'cut_Fair', 'cut_Good', 'cut_Ideal',
       'cut_Premium', 'cut_Very Good', 'color_D', 'color_E', 'color_F',
       'color_G', 'color_H', 'color_I', 'color_J', 'clarity_I1', 'clarity_IF',
       'clarity_SI1', 'clarity_SI2', 'clarity_VS1', 'clarity_VS2',
       'clarity_VVS1', 'clarity_VVS2']]
    y=df.price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model=model_function.fit(X_train, y_train)
    pred=model.predict(X_test)
    print("Mean Absolute Error is :", mean_absolute_error(y_test, pred))
    print('Mean Squared Error is :', mean_squared_error(y_test, pred))
    print('The R2 square value is :', r2_score(y_test, pred)*100)
    return model

def saveTrainedModel(model,modelname):
    filename = './models/{}.sav'.format(modelname)
    pickle.dump(model, open(filename, 'wb')) 




