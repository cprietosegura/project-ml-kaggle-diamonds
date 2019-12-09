import pickle
from src.cleaning import preparingDataset, preparingDatasetBis


def prepareSubmission(filepath,csv_name):
    loaded_model = pickle.load(open(filepath, 'rb'))
    test=preparingDataset('test.csv')
    X=test[['carat','depth', 'table', 'cut_Fair', 'cut_Good', 'cut_Ideal',
       'cut_Premium', 'cut_Very Good', 'color_D', 'color_E', 'color_F',
       'color_G', 'color_H', 'color_I', 'color_J', 'clarity_I1', 'clarity_IF',
       'clarity_SI1', 'clarity_SI2', 'clarity_VS1', 'clarity_VS2',
       'clarity_VVS1', 'clarity_VVS2']]
    test['price']=loaded_model.predict(X)
    test=test[['id','price']]
    return test.to_csv('./output/{}_2.csv'.format(csv_name), index=False)