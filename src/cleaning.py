import pandas as pd 

def preparingDataset(csv_path):
    df=pd.read_csv(csv_path)
    df=pd.get_dummies(df)
    df.drop(columns=['x','z','y'],inplace=True)
    return df

def preparingDatasetBis(csv_path):
    #hacemos el mismo proceso pero dejamos la feature z, finalmente obtengo mejores resultados retirando la z.
    df=pd.read_csv(csv_path)
    df=pd.get_dummies(df)
    df.drop(columns=['x','y'],inplace=True)
    return df

