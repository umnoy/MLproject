import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

from data import clean_titanic, to_onehot, CAT_FEATURES

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

y = train_data["Survived"]
X, X_test = clean_titanic(train_data, test_data)
X_encoded, X_test_encoded = to_onehot(X, X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

sklearn_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=1),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1),
}

results = []
for name, model in sklearn_models.items():
    scores = cross_val_score(model, X_encoded, y, cv=cv, scoring="accuracy")
    results.append({"model": name, "mean_acc": scores.mean(), "std": scores.std()})
    print(f"{name:20s} mean_acc={scores.mean():.4f}  std={scores.std():.4f}")


cb_scores = []
for train_idx, val_idx in cv.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, verbose=0, random_state=1)
    model.fit(X_tr, y_tr, cat_features=CAT_FEATURES)
    cb_scores.append(accuracy_score(y_val, model.predict(X_val)))

results.append({"model": "CatBoost", "mean_acc": np.mean(cb_scores), "std": np.std(cb_scores)})
print(f"{'CatBoost':20s} mean_acc={np.mean(cb_scores):.4f}  std={np.std(cb_scores):.4f}")

results_df = pd.DataFrame(results).sort_values("mean_acc", ascending=False)
results_df.to_csv("outputs/model_comparison.csv", index=False)
print("\nЛучшая модель:")
print(results_df.iloc[0])