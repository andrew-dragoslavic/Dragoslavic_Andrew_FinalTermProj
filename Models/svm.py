import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm

# from sklearn.datasets import load_breast_cancer
# import pandas as pd

# # Load the dataset
# data = load_breast_cancer()

# # Create a DataFrame
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df['target'] = data.target

data = pd.read_csv('diabetes.csv')

X = data.drop('Outcome', axis=1)
Y = data['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, Y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, y_pred))