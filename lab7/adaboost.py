import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

load_data = pd.read_csv('income.csv')
print(load_data.head())

# Separating predictors and response
X = load_data.iloc[:, 0:6]
y = load_data.iloc[:, 6]

# split dataset into training set and test set
# 75% training
# 25% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# n_estimators : the number of weak learners to train iteratively
# The more is your estimators the more trees will be build and used for voting.
AdaModel = AdaBoostClassifier(n_estimators=100,
                              learning_rate=1)

# Train Adaboost classifier
model = AdaModel.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = model.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))