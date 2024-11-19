import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, brier_score_loss, roc_curve, roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import svm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

def calculate_metrics(FP, FN, TP, TN):
    P = TP + FN
    N = TN + FP

    TPR = TP/P if P != 0 else 0
    TNR = TN/N if N != 0 else 0
    FPR = FP/N if N != 0 else 0
    FNR = FN/P if P != 0 else 0

    recall = TPR
    precision = TP/(TP+FP) if (TP+FP) != 0 else 0
    F1 = (2*TP)/(2*TP+FP+FN) if (2*TP+FP+FN) != 0 else 0
    accuracy = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) != 0 else 0
    error_rate = (FP+FN)/(TP+TN+FP+FN) if (TP+TN+FP+FN) != 0 else 0

    BACC = (TPR+TNR)/2
    TSS = (TP/(TP+FN))-(FP/(FP+TN))
    HSS = (2*((TP*TN)-(FP*FN)))/(((TP+FN)*(FN+TN)) + ((TP+FP)*(FP+TN)))

    return {
        'TPR': TPR,
        'TNR': TNR,
        'FPR': FPR,
        'FNR': FNR,
        'Recall': recall,
        'Precision': precision,
        'F1': F1,
        'Accuracy': accuracy,
        'Error Rate': error_rate,
        'BACC': BACC,
        'TSS': TSS,
        'HSS': HSS
    }

def random_forest(X_train, X_test, y_train, y_test):
    rf  = RandomForestClassifier()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    brier_score = brier_score_loss(y_test,y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    return {
        'y_pred': y_pred,
        'y_prob': y_prob,
        'brier_score': brier_score,
        'roc_auc': roc_auc
    }

def support_vector_machine(X_train, X_test, y_train, y_test):
    clf = svm.SVC(kernel='linear', probability=True) 
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    brier_score = brier_score_loss(y_test,y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)  

    return {
        'y_pred': y_pred,
        'y_prob': y_prob,
        'brier_score': brier_score,
        'roc_auc': roc_auc
    } 

def lstm(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Define the LSTM model with Input layer
    lstm_model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the LSTM model
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Predict and convert probabilities to binary labels
    y_pred_prob = lstm_model.predict(X_test)
    brier_score = brier_score_loss(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)          
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    return {
        'y_pred': y_pred,
        'y_prob': y_pred_prob,
        'brier_score': brier_score,
        'roc_auc': roc_auc
    }

def plot_roc(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guess
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def k_fold(X, Y, K, model):
    kf = KFold(n_splits=K, shuffle = True, random_state=42)
    metrics_list = []
    cumulative_cm = np.zeros((2, 2), dtype=int)
    all_y_true = []
    all_y_prob = []

    for i, (train_index, test_index) in enumerate(kf.split(X), start = 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        if model == 'Random Forest':
            rf = random_forest(X_train, X_test, y_train, y_test)
            y_pred = rf['y_pred']
            y_prob = rf['y_prob']
            brier_score = rf['brier_score']
            roc_auc = rf['roc_auc']

            all_y_true.extend(y_test)
            all_y_prob.extend(y_prob)
        elif model == 'SVM':
            clf = support_vector_machine(X_train, X_test, y_train, y_test)
            y_pred = clf['y_pred']
            y_prob = clf['y_prob']
            brier_score = clf['brier_score']
            roc_auc = clf['roc_auc']       
            
            all_y_true.extend(y_test)
            all_y_prob.extend(y_prob)
        elif model == 'LSTM':
            lstm_results = lstm(X_train,X_test, y_train, y_test)
            y_pred = lstm_results['y_pred']
            y_pred_prob = lstm_results['y_prob']
            brier_score = lstm_results['brier_score']
            roc_auc = lstm_results['roc_auc']

            all_y_true.extend(y_test)
            all_y_prob.extend(y_pred_prob)

        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        TP = cm[0, 0]
        FN = cm[0, 1]
        FP = cm[1, 0]
        TN = cm[1, 1]

        cumulative_cm += cm # Add current fold's confusion matrix to the cumulative matrix

        fold_metrics = calculate_metrics(FP,FN,TP,TN)
        fold_metrics['Fold'] = i
        fold_metrics['Brier Score'] = brier_score
        fold_metrics['AUC'] = roc_auc
        metrics_list.append(fold_metrics)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.set_index('Fold', inplace=True)

    avg_metrics = metrics_df.mean().to_dict()
    avg_metrics['Fold'] = 'Average'
    metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_metrics]).set_index('Fold')])

    print(f"Metrics Table Across All Folds for {model}:\n")
    print(metrics_df)

    disp = ConfusionMatrixDisplay(confusion_matrix=cumulative_cm, display_labels=[1,0])
    disp.plot(cmap='Blues')
    plt.title('Cumulative Confusion Matrix After All Folds')
    plt.show()

    roc_auc = roc_auc_score(all_y_true, all_y_prob)
    fpr, tpr, threshold = roc_curve(all_y_true, all_y_prob)

    plot_roc(fpr,tpr,roc_auc)

    return metrics_df

data = pd.read_csv("Data/heart.csv")
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)

X = data.drop('target', axis=1)
Y = data['target']

avg_rf = k_fold(X,Y,10, 'Random Forest')
avg_svm = k_fold(X,Y,10, 'SVM')
avg_lstm = k_fold(X,Y,10, 'LSTM')