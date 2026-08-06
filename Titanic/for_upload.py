import pandas as pd
from catboost import CatBoostClassifier
from data import clean_titanic, CAT_FEATURES

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

y = train_data["Survived"]
X, X_test = clean_titanic(train_data, test_data)
model = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, verbose=0, random_state=1)
model.fit(X, y, cat_features=CAT_FEATURES)
predictions = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predictions
})
submission.to_csv("outputs/submission.csv", index=False)
print(predictions[:5], predictions.dtype, predictions.shape)