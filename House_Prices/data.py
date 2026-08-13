import numpy as np
import pandas as pd


def prepare_data(train_path, test_path, id_column='Id', dummies=True):
    """
    все преобразования из ноутбука с eda
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index) #единственный пропуск в electrical
    df_train = df_train.drop(df_train[df_train['Id'].isin([1299, 524])].index) # выбросы по GrLivArea

    missing_counts = df_train.isnull().sum()
    cols_to_drop = missing_counts[missing_counts > 1].index.tolist()
    df_train = df_train.drop(columns=cols_to_drop)
    df_test = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])

    y_train = np.log(df_train['SalePrice'])
    df_train = df_train.drop(columns=['SalePrice'])

    ids = df_test[id_column] 
    df_train = df_train.drop(columns=[id_column])
    df_test = df_test.drop(columns=[id_column])

    df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
    df_test['GrLivArea'] = np.log(df_test['GrLivArea'])

    # HasBsmt + условный лог TotalBsmtSF
    for df in (df_train, df_test):
        df['TotalBsmtSF'] = df['TotalBsmtSF'].astype(float)
        df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
        mask = df['TotalBsmtSF'] > 0
        df.loc[mask, 'TotalBsmtSF'] = np.log(df.loc[mask, 'TotalBsmtSF'])


    numeric_cols = df_train.select_dtypes(include=[np.number]).columns 
    medians = df_train[numeric_cols].median()
    df_test[numeric_cols] = df_test[numeric_cols].fillna(medians) #заполняем медианами на случай если появляются нули в тесте после тех же преобразований

    categorical_cols = df_train.select_dtypes(include='object').columns
    modes = df_train[categorical_cols].mode().iloc[0]
    df_test[categorical_cols] = df_test[categorical_cols].fillna(modes)

    if dummies:
        combined = pd.concat([df_train, df_test], keys=['train', 'test'])
        combined = pd.get_dummies(combined) #совместное кодирование чтоб совпали признаки
        df_train = combined.xs('train')
        df_test = combined.xs('test')

    return df_train, y_train, df_test, list(categorical_cols), ids