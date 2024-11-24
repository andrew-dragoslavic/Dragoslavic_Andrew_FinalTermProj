import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, brier_score_loss, roc_curve, roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import svm
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler
from keras.src.optimizers import Adam

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

def plot_roc(fpr, tpr, roc_auc, model_name):
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guess
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

def eval_model(model_name, model_function, X_train, X_test, y_train, y_test, all_y_true, all_y_prob, cumulative_cm, metrics_dict, metrics_list, i):
    res = model_function(X_train, X_test, y_train, y_test)
    y_pred = res['y_pred']
    y_prob = res['y_prob']
    brier_score = res['brier_score']
    roc_auc = res['roc_auc']
    all_y_true.extend(y_test)
    all_y_prob.extend(y_prob)
    cm = confusion_matrix(y_test, y_pred, labels=[1,0])
    TP, FN, FP, TN = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    cumulative_cm += cm
    fold_metrics = calculate_metrics(FP,FN,TP,TN)
    fold_metrics['Brier Score'] = brier_score
    fold_metrics['AUC'] = roc_auc
    fold_metrics['Fold'] = i
    metrics_dict[model_name] = {key: value for key, value in fold_metrics.items() if key != 'Fold'}
    metrics_list.append(fold_metrics)

    return fold_metrics

def process_metrics_dataframe(metrics_list):
    df = pd.DataFrame(metrics_list)
    df["Fold"] = df["Fold"].astype(object)
    averages = df.mean(numeric_only=True)
    averages["Fold"] = "Average"  
    df = pd.concat([df, pd.DataFrame([averages])], ignore_index=True)
    df.set_index("Fold", inplace=True)
    return df

def k_fold(X, Y, K):
    kf = KFold(n_splits=K, shuffle = True, random_state=42)
    metrics_list_rf, metrics_list_clf, metrics_list_lstm = [], [], []
    metrics_dict = {}
    cumulative_cm_rf,cumulative_cm_clf, cumulative_cm_lstm = np.zeros((2, 2), dtype=int),np.zeros((2, 2), dtype=int),np.zeros((2, 2), dtype=int)
    all_y_true_rf, all_y_true_clf, all_y_true_lstm  = [], [], []
    all_y_prob_rf, all_y_prob_clf, all_y_prob_lstm = [], [], []

    for i, (train_index, test_index) in enumerate(kf.split(X), start = 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

        eval_model(
            model_name='Random Forest', model_function=random_forest,
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            all_y_true=all_y_true_rf, all_y_prob=all_y_prob_rf, cumulative_cm=cumulative_cm_rf,
            metrics_dict=metrics_dict, metrics_list=metrics_list_rf, i = i
        )

        eval_model(
            model_name='SVM', model_function=support_vector_machine,
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            all_y_true=all_y_true_clf, all_y_prob=all_y_prob_clf, cumulative_cm=cumulative_cm_clf,
            metrics_dict=metrics_dict, metrics_list=metrics_list_clf, i = i
        )

        eval_model(
            model_name='LSTM', model_function=lstm,
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            all_y_true=all_y_true_lstm, all_y_prob=all_y_prob_lstm, cumulative_cm=cumulative_cm_lstm,
            metrics_dict=metrics_dict, metrics_list=metrics_list_lstm, i = i
        )

        df = pd.DataFrame(metrics_dict)
        print(f"Fold {i}:\n{df}")

    df_rf = process_metrics_dataframe(metrics_list_rf)
    df_clf = process_metrics_dataframe(metrics_list_clf)
    df_lstm = process_metrics_dataframe(metrics_list_lstm)

    print(f"\nRandom Forest Metrics:\n{df_rf}")
    print(f"\nSVM Metrics:\n{df_clf}")
    print(f"\nLSTM Metrics:\n{df_lstm}")

    models = [
        ('Random Forest', all_y_true_rf, all_y_prob_rf, cumulative_cm_rf),
        ('SVM', all_y_true_clf, all_y_prob_clf, cumulative_cm_clf),
        ('LSTM', all_y_true_lstm, all_y_prob_lstm, cumulative_cm_lstm)
    ]

    for model_name, y_true, y_prob, cm in models:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        plot_roc(fpr,tpr, roc_auc, model_name)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,0])
        disp.plot(cmap='Blues')
        plt.title(f'Cumulative Confusion Matrix After All Folds - {model_name}')
        plt.show()

data = pd.read_csv("Data/heart.csv")
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)

X = data.drop('target', axis=1)
Y = data['target']

k_fold(X,Y,10)