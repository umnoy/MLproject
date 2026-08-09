import pandas as pd

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Initial"]
CAT_FEATURES = ["Sex", "Embarked", "Initial"]

#средние в ноутбуке (groupby('Initial')['Age'].mean())
AGE_BY_INITIAL = {"Mr": 33, "Mrs": 36, "Master": 5, "Miss": 22, "Other": 46}

RARE_TITLES_MAP = {
    "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss",
    "Dr": "Mr", "Major": "Mr", "Capt": "Mr", "Sir": "Mr", "Don": "Mr",
    "Lady": "Mrs", "Countess": "Mrs",
    "Jonkheer": "Other", "Col": "Other", "Rev": "Other",
}


def _extract_initial(df):
    df = df.copy()
    df["Initial"] = df["Name"].str.extract(r"([A-Za-z]+)\.")
    df["Initial"] = df["Initial"].replace(RARE_TITLES_MAP)
    return df


def _fill_age(df):
    df = df.copy()
    for initial, age in AGE_BY_INITIAL.items():
        mask = df["Age"].isnull() & (df["Initial"] == initial)
        df.loc[mask, "Age"] = age
    df["Age"] = df["Age"].fillna(df["Age"].median())  # на случай нового титула в test
    return df


def clean_titanic(train_df, test_df):
    train_df = _extract_initial(train_df)
    test_df = _extract_initial(test_df)
    train_df = _fill_age(train_df)
    test_df = _fill_age(test_df)

    embarked_mode = train_df["Embarked"].mode()[0]#заполнение модой по портам отправления
    train_df["Embarked"] = train_df["Embarked"].fillna(embarked_mode)
    test_df["Embarked"] = test_df["Embarked"].fillna(embarked_mode)

    fare_median = train_df["Fare"].median()
    test_df["Fare"] = test_df["Fare"].fillna(fare_median)

    return train_df[FEATURES], test_df[FEATURES]


def to_onehot(X_train, X_test):
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)  # выравниваем колонки
    return X_train, X_test

def prepare_titanic_data(train_path, test_path, column):
    train_df, test_df = load_data(train_path, test_path)
    
    y_train = train_df["Survived"]
    X_train_raw, X_test_raw = clean_titanic(train_df, test_df)
    
    X_train, X_test = to_onehot(X_train_raw, X_test_raw)
    
    return X_train, y_train, X_test, test_df[column]