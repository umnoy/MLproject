import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from data import preprocess_titanic

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

y = train_data["Survived"]
X, X_test = preprocess_titanic(train_data, test_data)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=1),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1),
}

results = []
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    results.append({"model": name, "mean_acc": scores.mean(), "std": scores.std()})
    print(f"{name:20s} mean_acc={scores.mean():.4f}  std={scores.std():.4f}")

results_df = pd.DataFrame(results).sort_values("mean_acc", ascending=False)
results_df.to_csv("outputs/model_comparison.csv", index=False)
print("\nЛучшая модель:")
print(results_df.iloc[0])