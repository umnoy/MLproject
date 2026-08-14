import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from data import prepare_data


RANDOM_STATE = 1
N_FOLDS = 5
#параметры бустингов украл у автора одного из анализов
ELASTIC_PARAMS = dict(alpha=0.0005, l1_ratio=0.9, random_state=RANDOM_STATE)
LASSO_PARAMS = dict(alpha=0.0005, random_state=RANDOM_STATE)
KRR_PARAMS = dict(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBR_PARAMS = dict(n_estimators=3000, learning_rate=0.05, max_depth=4,
                   max_features='sqrt', min_samples_leaf=15, min_samples_split=10,
                   loss='huber', random_state=RANDOM_STATE)
XGB_PARAMS = dict(n_estimators=2200, learning_rate=0.05, max_depth=3,
                   subsample=0.7, colsample_bytree=0.7, random_state=RANDOM_STATE)
LGB_PARAMS = dict(n_estimators=720, learning_rate=0.05, num_leaves=5,
                   min_child_samples=6, random_state=RANDOM_STATE)


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(m) for m in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=N_FOLDS):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=RANDOM_STATE)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([np.column_stack([model.predict(X) for model in base_models]).mean(axis=1) for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


def rmse_cv(model, X, y, cv):
    scores = -cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv)
    return np.sqrt(scores)


def main():
    X_train_df, y_train, X_test_df, cat_features, test_ids = prepare_data("data/train.csv", "data/test.csv", id_column="Id", dummies=True, scale=True)
    X = X_train_df.values
    y = y_train.values
    X_test = X_test_df.values

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    ENet = ElasticNet(**ELASTIC_PARAMS)
    lasso = Lasso(**LASSO_PARAMS)
    KRR = KernelRidge(**KRR_PARAMS)
    GBoost = GradientBoostingRegressor(**GBR_PARAMS)


    print("усреднение ENet + GBoost + KRR + Lasso")
    averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))
    score = rmse_cv(averaged_models, X, y, cv)
    print(f"rmse: {score.mean():.5f} ({score.std():.5f})")

 
    print("стекинг с метой: ENet + GBoost + KRR + meta")
    stacked_averaged_models = StackingAveragedModels(
        base_models=(ENet, GBoost, KRR),
        meta_model=lasso,
        n_folds=N_FOLDS,
    )
    score = rmse_cv(stacked_averaged_models, X, y, cv)
    print(f"rmse: {score.mean():.5f} ({score.std():.5f})")


    print("XGBoost и LightGBM отдельно")
    model_xgb = XGBRegressor(**XGB_PARAMS)
    score = rmse_cv(model_xgb, X, y, cv)
    print(f"XGBoost score: {score.mean():.5f} ({score.std():.5f})")

    model_lgb = LGBMRegressor(**LGB_PARAMS)
    score = rmse_cv(model_lgb, X, y, cv)
    print(f"LightGBM score: {score.mean():.5f} ({score.std():.5f})")

 
    print("еще раз обучаем и строим ансамбль")

    stacked_averaged_models.fit(X, y)
    stacked_train_pred = stacked_averaged_models.predict(X)
    stacked_pred = stacked_averaged_models.predict(X_test)
    print(f"RMSE стекинга на train: "
          f"{np.sqrt(np.mean((y - stacked_train_pred) ** 2)):.5f}")

    model_xgb.fit(X, y)
    xgb_train_pred = model_xgb.predict(X)
    xgb_pred = model_xgb.predict(X_test)
    print(f"RMSE XGBoost на train: "
          f"{np.sqrt(np.mean((y - xgb_train_pred) ** 2)):.5f}")

    model_lgb.fit(X, y)
    lgb_train_pred = model_lgb.predict(X)
    lgb_pred = model_lgb.predict(X_test)
    print(f"RMSE LightGBM на train: "
          f"{np.sqrt(np.mean((y - lgb_train_pred) ** 2)):.5f}")

    ensemble_train_pred = stacked_train_pred * 0.70 + xgb_train_pred * 0.15 + lgb_train_pred * 0.15
    ensemble_rmse = np.sqrt(np.mean((y - ensemble_train_pred) ** 2))
    print(f"RMSE финального ансамбля на train: {ensemble_rmse:.5f}")

    ensemble_pred = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15

    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": np.exp(ensemble_pred),
    })
    submission.to_csv("outputs/submission_ensemble.csv", index=False)
    print("все")


if __name__ == "__main__":
    main()