import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('train_cleaned.csv')

X = data.drop('Exited', axis=1)
Y = data['Exited']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = GaussianNB()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)
