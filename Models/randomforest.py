import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def k_fold(X, Y, K):
    kf = KFold(n_splits=K, shuffle = True, random_state=42)
    metrics_list = []
    cumulative_cm = np.zeros((2, 2), dtype=int)
    
    for i, (train_index, test_index) in enumerate(kf.split(X), start = 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        rf  = RandomForestClassifier()
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        TP = cm[0, 0]
        FN = cm[0, 1]
        FP = cm[1, 0]
        TN = cm[1, 1]

        cumulative_cm += cm # Add current fold's confusion matrix to the cumulative matrix

        fold_metrics = calculate_metrics(FP,FN,TP,TN)
        fold_metrics['Fold'] = i
        metrics_list.append(fold_metrics)

        # print(f"Fold {i} Metrics")
        # for metric, value in fold_metrics.items():
        #     print(f' {metric}: {value:.4f}')
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.set_index('Fold', inplace=True)

    avg_metrics = metrics_df.mean().to_dict()
    avg_metrics['Fold'] = 'Average'
    metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_metrics]).set_index('Fold')])

    print("Metrics Table Across All Folds:\n")
    print(metrics_df)

    disp = ConfusionMatrixDisplay(confusion_matrix=cumulative_cm, display_labels=[0, 1])
    disp.plot(cmap='Blues')
    plt.title('Cumulative Confusion Matrix After All Folds')
    plt.show()

    return metrics_df

def calculate_metrics(FP, FN, TP, TN):
    P = TP + FN
    N = TN + FP

    TPR = TP/P if P != 0 else 0
    TNR = TN/N if N != 0 else 0
    FPR = FP/N if N != 0 else 0
    FNR = FN/P if P != 0 else 0

    recall = TPR
    precision = TP/(TP+FP) if (TP+FP) != 0 else 0
    F1 = (2*TP)/(2*TP+FP+FN)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    error_rate = (FP+FN)/(TP+TN+FP+FN)

    return {
        'TPR': TPR,
        'TNR': TNR,
        'FPR': FPR,
        'FNR': FNR,
        'Recall': recall,
        'Precision': precision,
        'F1': F1,
        'Accuracy': accuracy,
        'Error Rate': error_rate
    }

data = pd.read_csv("Data/diabetes.csv")

X = data.drop('Outcome', axis=1)
Y = data['Outcome']

avg = k_fold(X,Y,10)
