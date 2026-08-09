import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

mlp_oof = np.load("outputs/scheduler4/mlp_oof.npy")
mlp_test = np.load("outputs/scheduler4/mlp_test_preds.npy")

cb_oof = np.load("outputs/not_nn/catboost_oof.npy")
cb_test = np.load("outputs/not_nn/catboost_test_preds.npy")

lgb_oof = np.load("outputs/not_nn/lgb_oof.npy")
lgb_test = np.load("outputs/not_nn/lgb_test_preds.npy")


train_data = pd.read_csv("data/train.csv")
y = train_data["Survived"].values

#усреднение
avg_oof = (mlp_oof + cb_oof + lgb_oof) / 3
avg_acc = accuracy_score(y, (avg_oof > 0.5).astype(int))
print(f"Averaging OOF accuracy: {avg_acc:.4f}")

avg_test = (mlp_test + cb_test + lgb_test) / 3

#Voting
mlp_class = (mlp_oof > 0.5).astype(int)
cb_class = (cb_oof > 0.5).astype(int)
lgb_class = (lgb_oof > 0.5).astype(int)

votes = mlp_class + cb_class + lgb_class
hard_voting_oof = (votes >= 2).astype(int)

voting_acc = accuracy_score(y, hard_voting_oof)
print(f"Voting OOF accuracy: {voting_acc:.4f}")

#stacking
cv_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

stack_features = np.column_stack([mlp_oof, cb_oof, lgb_oof])
stack_oof = np.zeros(len(y))

for train_idx, val_idx in cv_meta.split(stack_features, y):
    X_tr, X_val = stack_features[train_idx], stack_features[val_idx]
    y_tr = y[train_idx]

    meta_model = LogisticRegression()
    meta_model.fit(X_tr, y_tr)
    stack_oof[val_idx] = meta_model.predict_proba(X_val)[:, 1]

stack_acc = accuracy_score(y, (stack_oof > 0.5).astype(int))
print(f"Stacking OOF accuracy: {stack_acc:.4f}")

test_data = pd.read_csv("data/test.csv")

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": (avg_test > 0.5).astype(int)
})
submission.to_csv("outputs/ensemble_submission.csv", index=False)