import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import time

from data import prepare_data


train_data_path = "data/train.csv"
test_data_path = "data/test.csv"
save_dir = os.path.join("outputs", "not_nn")
os.makedirs(save_dir, exist_ok=True)


X, y, X_test, CAT_FEATURES, IDs = prepare_data(train_data_path, test_data_path, dummies=False)
X_encoded, y, X_test_encoded, _, IDs = prepare_data(train_data_path, test_data_path, dummies=True)

cv = KFold(n_splits=5, shuffle=True, random_state=1)


def rmse_cv_score(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return -scores.mean()

def rmse_cv_score_catboost(params, X, y, cat_features, cv):
    fold_rmses = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostRegressor(**params, verbose=0)
        model.fit(X_tr, y_tr, cat_features=cat_features)
        pred = model.predict(X_val)
        fold_rmses.append(np.sqrt(mean_squared_error(y_val, pred)))
    return np.mean(fold_rmses)

def rmse_cv_score_lightgbm(params, X, y, cat_features, cv):
    fold_rmses = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for col in cat_features:
            X_tr[col] = X_tr[col].astype("category")
            X_val[col] = X_val[col].astype("category")

        model = LGBMRegressor(**params, verbose=-1)
        model.fit(X_tr, y_tr, categorical_feature=cat_features)
        pred = model.predict(X_val)
        fold_rmses.append(np.sqrt(mean_squared_error(y_val, pred)))
    return np.mean(fold_rmses)


def greed_search(model_class, param_grid, fixed_params, X, y, cv, catboost = False, light = False):
    best_params = dict(fixed_params)

    model = model_class(**best_params)
    if catboost:
        best_score = rmse_cv_score_catboost(best_params, X, y, CAT_FEATURES, cv)
    elif light:
        best_score = rmse_cv_score_lightgbm(best_params, X, y, CAT_FEATURES, cv)
    else:
        best_score = rmse_cv_score(model, X, y, cv)

    for param_name, values in param_grid.items():
        for value in values:
            params = {**best_params, param_name: value}
            model = model_class(**params)
            if catboost:
                score = rmse_cv_score_catboost(params, X, y, CAT_FEATURES, cv)
            elif light:
                score = rmse_cv_score_lightgbm(params, X, y, CAT_FEATURES, cv)
            else:
                score = rmse_cv_score(model, X, y, cv)


            if score < best_score:
                best_params[param_name] = value
                best_score = score


    return best_params, best_score

param_grid = {
    "Ridge": {
        "alpha": [0.1, 1.0, 10.0, 50.0, 100.0],
    },
    "Lasso": {
        "alpha": [0.0005, 0.001, 0.005, 0.01, 0.05],
    },
    "ElasticNet": {
        "alpha": [0.0005, 0.001, 0.01, 0.1],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 10, 15],
    },
    "DecisionTree": {
        "max_depth": [3, 5, 7, 10, None],
    },
    "RandomForest": {
        "n_estimators": [100, 200, 400],
        "max_depth": [5, 10, 15, None],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
    },
}

sklearn_model_classes = {
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "KNN": KNeighborsRegressor,
    "DecisionTree": DecisionTreeRegressor,
    "RandomForest": RandomForestRegressor,
    "XGBoost": XGBRegressor,
}

sklearn_fixed_params = {
    "Ridge": {},
    "Lasso": {},
    "ElasticNet": {},
    "KNN": {},
    "DecisionTree": {"random_state": 1},
    "RandomForest": {"random_state": 1},
    "XGBoost": {"random_state": 1},
}
catboost_fixed_params = {"random_state": 1}
lightgbm_fixed_params = {"random_state": 1}

results = []

start_time = time.perf_counter()
for name in param_grid:
    best_params, best_score = greed_search(sklearn_model_classes[name], param_grid[name],sklearn_fixed_params[name], X_encoded, y, cv)
    results.append({"model": name, "rmse": best_score, "best_params": best_params})
    print(f"{name}:  rmse={best_score}  params={best_params}")


catboost_param_grid = {
    "iterations": [100, 300, 500],
    "depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
}

lightgbm_param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
}

best_params, best_score = greed_search(CatBoostRegressor, catboost_param_grid, catboost_fixed_params,X, y, cv, catboost=True)
results.append({"model": "CatBoost", "rmse": best_score, "best_params": best_params})
print(f"CatBoost:  rmse={best_score}  params={best_params}")

results_df = pd.DataFrame(results).sort_values("rmse")
results_df.to_csv(f"{save_dir}/model_comparison.csv", index=False)

print()
print("Лучшая модель:")
print(results_df.iloc[0])

best_row = results_df.iloc[0]
best_model_name = best_row["model"]
best_params = best_row["best_params"]

if best_model_name in sklearn_model_classes:
    model_class = sklearn_model_classes[best_model_name]
    model = model_class(**best_params)
    model.fit(X_encoded, y)
    predictions_log = model.predict(X_test_encoded)

elif best_model_name == "CatBoost":
    model = CatBoostRegressor(**best_params, verbose=0)
    model.fit(X, y, cat_features=CAT_FEATURES)
    predictions_log = model.predict(X_test)

elif best_model_name == "LightGBM":
    X_final = X.copy()
    X_test_final = X_test.copy()
    for col in CAT_FEATURES:
        X_final[col] = X_final[col].astype("category")
        X_test_final[col] = X_test_final[col].astype("category")

    model = LGBMRegressor(**best_params, verbose=-1)
    model.fit(X_final, y, categorical_feature=CAT_FEATURES)
    predictions_log = model.predict(X_test_final)

predictions = np.exp(predictions_log)

submission = pd.DataFrame({
    "Id": IDs,
    "SalePrice": predictions
})
submission.to_csv(f"{save_dir}/submission.csv", index=False)
print(predictions[:5])



end_time = time.perf_counter()
execution_time = end_time - start_time
minutes, seconds = divmod(execution_time, 60)
print(f"Execution time: {int(minutes)}m {seconds:.2f}s")