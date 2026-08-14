import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler


# колонки где nan это отсутствие и признак категориальный
NONE_FILL_CATEGORICAL = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType',
]

# то же самое, но признак числовой и поменяется на 0
ZERO_FILL_NUMERIC = [
    'GarageYrBlt', 'GarageArea', 'GarageCars',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath',
    'MasVnrArea',
]

#очень редкие пропуски просто заполнят медианы
MODE_FILL = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st',
             'Exterior2nd', 'SaleType', 'Functional', 'Utilities']

SKEW_THRESHOLD = 0.75


def prepare_data(train_path, test_path, id_column='Id', dummies=True, scale=False):
    """
    все преобразования из ноутбука
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    test_ids = df_test[id_column]

    df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)  # единственный пропуск Electrical
    df_train = df_train.drop(df_train[df_train['Id'].isin([1299, 524])].index)  # выбросы по GrLivArea

    y_train = np.log(df_train['SalePrice'])
    df_train = df_train.drop(columns=['SalePrice'])

    df_train = df_train.drop(columns=[id_column])
    df_test = df_test.drop(columns=[id_column])

    for col in NONE_FILL_CATEGORICAL:
        if col in df_train.columns:
            df_train[col] = df_train[col].fillna('None')
            df_test[col] = df_test[col].fillna('None')

    for col in ZERO_FILL_NUMERIC:
        if col in df_train.columns:
            df_train[col] = df_train[col].fillna(0)
            df_test[col] = df_test[col].fillna(0)

    #LotFrontage медианой по Neighborhood
    if 'LotFrontage' in df_train.columns:
        neighborhood_medians = df_train.groupby('Neighborhood')['LotFrontage'].median()
        global_median = df_train['LotFrontage'].median()  # запасной вариант, если в test встретится район, которого не было в train

        df_train['LotFrontage'] = df_train['LotFrontage'].fillna(
            df_train['Neighborhood'].map(neighborhood_medians)
        )
        df_test['LotFrontage'] = df_test['LotFrontage'].fillna(
            df_test['Neighborhood'].map(neighborhood_medians)
        )
        df_test['LotFrontage'] = df_test['LotFrontage'].fillna(global_median)

    for col in MODE_FILL: #для редких случайных пропусков
        if col in df_train.columns:
            mode_val = df_train[col].mode().iloc[0]
            df_train[col] = df_train[col].fillna(mode_val)
            df_test[col] = df_test[col].fillna(mode_val)

    for df in (df_train, df_test):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] #новый признак

    df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
    df_test['GrLivArea'] = np.log(df_test['GrLivArea'])

    for df in (df_train, df_test):
        df['TotalBsmtSF'] = df['TotalBsmtSF'].astype(float) #новый признак 
        df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
        mask = df['TotalBsmtSF'] > 0
        df.loc[mask, 'TotalBsmtSF'] = np.log(df.loc[mask, 'TotalBsmtSF'])


    numeric_cols_all = df_train.select_dtypes(include=[np.number]).columns
    already_handled = {'GrLivArea', 'TotalBsmtSF', 'HasBsmt'}
    candidate_cols = [c for c in numeric_cols_all if c not in already_handled]

    skewness = df_train[candidate_cols].apply(lambda x: skew(x.dropna()))
    skewed_cols = skewness[abs(skewness) > SKEW_THRESHOLD].index.tolist()

    for col in skewed_cols:
        if (df_train[col] >= 0).all() and (df_test[col].dropna() >= 0).all(): #перед лоагрифмированием на всякий проверка >0
            df_train[col] = np.log1p(df_train[col])
            df_test[col] = np.log1p(df_test[col])

    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    medians = df_train[numeric_cols].median()
    df_test[numeric_cols] = df_test[numeric_cols].fillna(medians)

    categorical_cols = df_train.select_dtypes(include='object').columns
    modes = df_train[categorical_cols].mode().iloc[0]
    df_test[categorical_cols] = df_test[categorical_cols].fillna(modes)

    cat_features = list(categorical_cols)

    if scale:
        scaler = StandardScaler()
        df_train[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
        df_test[numeric_cols] = scaler.transform(df_test[numeric_cols])

    if dummies:
        combined = pd.concat([df_train, df_test], keys=['train', 'test'])  # кодируем train и test вместе, чтобы колонки one-hot совпали
        combined = pd.get_dummies(combined)
        df_train = combined.xs('train')
        df_test = combined.xs('test')
        cat_features = []

    return df_train, y_train, df_test, cat_features, test_ids