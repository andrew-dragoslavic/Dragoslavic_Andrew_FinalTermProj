import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

data = pd.read_csv('train_cleaned.csv')

X = data.drop('Exited', axis=1)
Y = data['Exited']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

Y_pred = rf.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)