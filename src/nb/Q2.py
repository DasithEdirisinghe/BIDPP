#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

dir = '../Data/'
processed_data_dir = 'processed_data'

def read_file(feature):
    predictor = feature
    paac_train = pd.read_csv(f'{dir+processed_data_dir}/TR_{feature}.csv')
    paac_test = pd.read_csv(f'{dir+processed_data_dir}/TS_{feature}.csv')
    return predictor, paac_train, paac_test

# Need to balance the target variable. Oversampling is used
def balance(df):
    conditions = [
        (df['label'] == 0),
        (df['label'] == 1)
    ]

    values = [0 , 1]

    outcomes = np.select(conditions, values)

    rov = RandomOverSampler(random_state = 3)
    df_bal, out_bal = rov.fit_resample(df, outcomes)
    return df_bal

def feature_importance(X,Y):
    #Train a random forest regressor on the entire dataset
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, Y)

    # Get the feature importances from the trained random forest
    importances = rf.feature_importances_

    # Rank the features by importance and print the top 10
    sorted_idx = importances.argsort()[::-1]
    # # Select the top k features based on importance
    k = int((70/100)*len(sorted_idx))
    top_k_idx = sorted_idx[:k]
    X_top_k = X.iloc[:, top_k_idx]
    return X_top_k


class BinaryClassifier:
    def __init__(self, models):
        self.models = models
        self.best_model = None

    def train(self, X_train, y_train):
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model['model'].fit(X_train, y_train)
            print(f"{model_name} training completed.")

    # def evaluate(self, X_test, y_test):
    #     for model_name, model in self.models.items():
    #         print(f"Evaluating {model_name}...")
    #         y_pred = model['model'].predict(X_test)
    #         report = classification_report(y_test, y_pred)
    #         print(f"Classification Report for {model_name}:")
    #         print(report)

    def grid_search(self, X_train, y_train, param_grid, cv=5):
        for model_name, model in self.models.items():
            print(f"Performing Grid Search for {model_name}...")
            grid_search = GridSearchCV(model['model'], param_grid[model_name], cv=cv, scoring='f1')
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            print(f"Best parameters for {model_name}: {best_params}")
            print(f"Best score for {model_name}: {best_score}")
            self.models[model_name]['best_param'] = best_params
            self.models[model_name]['best_score'] = best_score
            self.models[model_name]['model'] = grid_search.best_estimator_
            if self.best_model is None or grid_search.best_score_ > self.best_model['score']:
                self.best_model = {'model_name': model_name, 'model': grid_search.best_estimator_, 'score': grid_search.best_score_, 'params': best_params}

    def get_best_model(self):
        return self.best_model


    # def plot_confusion_matrix(self, X_test, y_test):
    #     for model_name, model in self.models.items():
    #         y_pred = model['model'].predict(X_test)
    #         cm = confusion_matrix(y_test, y_pred)
    #         plt.figure(figsize=(6, 4))
    #         sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    #         plt.title(f'Confusion Matrix - {model_name}')
    #         plt.xlabel('Predicted')
    #         plt.ylabel('Actual')
    #         plt.show()

    # def compare_models(self, X_test, y_test):
    #     models = list(self.models.keys())
    #     metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    #     scores = np.zeros((len(models), len(metrics)))

    #     for i, (model_name, model) in enumerate(self.models.items()):
    #         y_pred = model['model'].predict(X_test)
    #         accuracy = accuracy_score(y_test, y_pred)
    #         precision = precision_score(y_test, y_pred)
    #         recall = recall_score(y_test, y_pred)
    #         f1 = f1_score(y_test, y_pred)
    #         scores[i] = [accuracy, precision, recall, f1]

    #     sns.set(style='whitegrid')
    #     plt.figure(figsize=(12, 6))
    #     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    #     bar_width = 0.15
    #     index = np.arange(len(models))

    #     for i, metric in enumerate(metrics):
    #         plt.bar(index + i * bar_width, scores[:, i], bar_width, color=colors[i], label=metric)

    #     plt.xlabel('Models')
    #     plt.ylabel('Score')
    #     plt.title(f'Model Comparison - {predictor}')
    #     plt.xticks(index + bar_width * (len(metrics) - 1) / 2, models)
    #     plt.legend()

    #     # Add data labels
    #     label_offset = 0.02
    #     for i, v in enumerate(index):
    #         for j, metric in enumerate(metrics):
    #             plt.text(v + j * bar_width, scores[i, j] + label_offset, f'{scores[i, j]:.3f}', ha='center', color='black', fontsize=8)

    #     plt.tight_layout()
    #     plt.show()
        
def report(y_true, y_pred):
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate F1 score
    f1 = f1_score(y_true, y_pred)

    # Calculate sensitivity (recall)
    sensitivity = recall_score(y_true, y_pred)

    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    # Calculate precision
    precision = precision_score(y_true, y_pred)

    weights = {'Accuracy': 0.3, 'F1': 0.2, 'Sensitivity': 0.2, 'Specificity': 0.1, 'Precision': 0.2}

    data = {'Accuracy': [accuracy],
            'F1': [f1],
            'Sensitivity': [sensitivity],
            'Specificity': [specificity],
            'Precision': [precision]}
    df = pd.DataFrame(data)

    # Calculate the combined score
    combined_score = sum([weights[metric] * df[metric].values[0] for metric in weights])
    df['Combined Score'] = combined_score

    print(df)
    df.to_csv(f'../out/classification_metrics.csv', index=False)

def output_results(predict_results):
    results = pd.DataFrame({'predicted_values': predict_results})
    results = pd.concat([paac_test['id'], results], axis=1)

    results['Numeric'] = results['id'].str.split('_').str[1].astype(int)

    predictions_neg = results[results['id'].str.startswith('N')]
    # Sort the DataFrame based on the 'Numeric' column in ascending order
    predictions_neg = predictions_neg.sort_values('Numeric')

    # Remove the 'Numeric' column from the sorted DataFrame
    predictions_neg = predictions_neg.drop('Numeric', axis=1)

    predictions_pos = results[results['id'].str.startswith('P')]
    # Sort the DataFrame based on the 'Numeric' column in ascending order
    predictions_pos = predictions_pos.sort_values('Numeric')

    # Remove the 'Numeric' column from the sorted DataFrame
    predictions_pos = predictions_pos.drop('Numeric', axis=1)


    predictions_pos.to_csv(f'../out/predictions_pos.txt', index=False, header=True, sep='\t')
    predictions_neg.to_csv(f'../out/predictions_neg.txt', index=False, header=True, sep='\t')


'''
Read Files
'''
predictor, paac_train, paac_test = read_file('AAC')

'''
Balance the Dataset
'''
paac_train_bal = balance(paac_train)


X = paac_train.drop(['id', 'label'], axis=1, inplace=False)
Y = paac_train['label']

'''
Feature Importance
'''
X_top_k = feature_importance(X,Y)
preprocessed = pd.concat([paac_train['label'], X_top_k], axis=1)
preprocessed = pd.concat([paac_train['id'], preprocessed], axis=1)

X = preprocessed.drop(['id', 'label'], axis=1, inplace=False)
Y = preprocessed['label']

'''
Split the Dataset
'''
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2)

important_cols = X.columns

'''
Scale the Dataset
'''
scaler = StandardScaler()
normal = ColumnTransformer([('normalize', scaler, important_cols)], remainder = 'passthrough')
X_train = pd.DataFrame(normal.fit_transform(X_train), columns = important_cols)
X_val = pd.DataFrame(normal.transform(X_val), columns = important_cols)

joblib.dump(normal, f'../out/{predictor}_normal.joblib')

models = {
    'SVM': {
        'model': SVC(C=1, kernel='rbf'),
        'best_param': None,
        'best_score': None
        },
    'CatBoost': {
        'model': CatBoostClassifier(),
        'best_param': None,
        'best_score': None
        }
}


'''
Train the Model
'''
pipe = Pipeline([('scaler', normal), ('model', models['SVM']['model'])])
pipe.fit(X_train, Y_train)
joblib.dump(pipe, f"../out/{predictor}_model.joblib")


'''
Load data
'''
predictor_model = joblib.load(f"../out/{predictor}_model.joblib")
normal = joblib.load(f'../out/{predictor}_normal.joblib')


'''
Predict the Test dataset
'''
features = paac_test.drop(['id', 'label'], axis=1, inplace = False)
scaled_features = pd.DataFrame(normal.transform(features), columns = important_cols)
predict_results = pipe.predict(scaled_features)
report(paac_test.label, predict_results)


'''
Output the results to a txt file
'''
output_results(predict_results)










