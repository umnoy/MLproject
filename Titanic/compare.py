import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier



from data import clean_titanic, to_onehot, CAT_FEATURES

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
save_dir = os.path.join("outputs", "not_nn")
os.makedirs(save_dir, exist_ok=True)

y = train_data["Survived"]
X, X_test = clean_titanic(train_data, test_data)
X_encoded, X_test_encoded = to_onehot(X, X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

sklearn_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LogisticRegression_L2": LogisticRegression(max_iter=1000, penalty="l2", solver="liblinear"),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=1),
    "RandomForest": RandomForestClassifier(n_estimators=170, max_depth=5, random_state=1),
    "XGBoost" : XGBClassifier(n_estimators=170, max_depth=5, learning_rate=0.1, random_state=1, eval_metric="logloss")
}

results = []
for name, model in sklearn_models.items():
    scores = cross_val_score(model, X_encoded, y, cv=cv, scoring="accuracy")
    results.append({"model": name, "mean_acc": scores.mean(), "std": scores.std()})
    print(f"{name:20s} mean_acc={scores.mean():.4f}  std={scores.std():.4f}")


cb_oof = np.zeros(len(y))
cb_test_preds = np.zeros(len(X_test))
cb_fold_accs = []

for train_idx, val_idx in cv.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, verbose=0, random_state=1)
    model.fit(X_tr, y_tr, cat_features=CAT_FEATURES)

    val_proba = model.predict_proba(X_val)[:, 1]
    cb_oof[val_idx] = val_proba
    cb_test_preds += model.predict_proba(X_test)[:, 1] / cv.get_n_splits()

    fold_acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))
    cb_fold_accs.append(fold_acc)

cb_oof_acc = accuracy_score(y, (cb_oof > 0.5).astype(int))
results.append({
    "model": "CatBoost",
    "mean_acc": np.mean(cb_fold_accs),
    "std": np.std(cb_fold_accs),
    "oof_acc": cb_oof_acc,
})
print(f"{'CatBoost':20s} CV_acc={np.mean(cb_fold_accs):.4f}  std = {np.std(cb_fold_accs):.4f}  OOF_acc={cb_oof_acc:.4f}")
np.save(f"{save_dir}/catboost_oof.npy", cb_oof)
np.save(f"{save_dir}/catboost_test_preds.npy", cb_test_preds)



X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in CAT_FEATURES:
    X_lgb[col] = X_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

lgb_oof = np.zeros(len(y))
lgb_test_preds = np.zeros(len(X_test))
lgb_fold_accs = []

for train_idx, val_idx in cv.split(X_lgb, y):
    X_tr, X_val = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(n_estimators=170, max_depth=6, learning_rate=0.1, random_state=1, verbose=-1)
    model.fit(X_tr, y_tr, categorical_feature=CAT_FEATURES)
    
    val_proba = model.predict_proba(X_val)[:, 1]
    lgb_oof[val_idx] = val_proba
    lgb_test_preds += model.predict_proba(X_test_lgb)[:, 1] / cv.get_n_splits()

    fold_acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))
    lgb_fold_accs.append(fold_acc)

lgb_oof_acc = accuracy_score(y, (lgb_oof > 0.5).astype(int))
results.append({"model": "LightGBM", "mean_acc": np.mean(lgb_fold_accs), "std": np.std(lgb_fold_accs), "oof_acc": lgb_oof_acc})
print(f"{'LightGBM':20s} mean_acc={np.mean(lgb_fold_accs):.4f}  std={np.std(lgb_fold_accs):.4f}")
np.save(f"{save_dir}/lgb_oof.npy", lgb_oof)
np.save(f"{save_dir}/lgb_test_preds.npy", lgb_test_preds)

results_df = pd.DataFrame(results).sort_values("mean_acc", ascending=False)
results_df.to_csv(f"{save_dir}/model_comparison.csv", index=False)
print("\nЛучшая модель:")
print(results_df.iloc[0])