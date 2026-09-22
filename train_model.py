"""
Script untuk melatih dan menyimpan model dropout siswa.
Jalankan di Colab atau lokal, lalu push model/best_model.pkl ke GitHub.

Usage:
    python train_model.py
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

DATA_URL = (
    'https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset'
    '/main/students_performance/data.csv'
)

print('Loading data...')
df = pd.read_csv(DATA_URL, sep=';')
df = df.dropna(subset=['Status'])

# Biner: Dropout (1) vs Graduate (0)
df = df[df['Status'].isin(['Dropout', 'Graduate'])].copy()
df['Target'] = (df['Status'] == 'Dropout').astype(int)

X = df.drop(columns=['Status', 'Target'])
y = df['Target']

print(f'Samples: {len(df)} | Dropout rate: {y.mean():.2%}')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

print('Training Gradient Boosting...')
model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1,
    max_depth=4, subsample=0.8, random_state=42
)
model.fit(X_train_scaled, y_train_res)

from sklearn.metrics import accuracy_score, roc_auc_score
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]
print(f'Accuracy : {accuracy_score(y_test, y_pred):.4f}')
print(f'ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}')

os.makedirs('model', exist_ok=True)
pipeline = Pipeline([('scaler', scaler), ('model', model)])
joblib.dump(pipeline, 'model/best_model.pkl')
print('Model saved to model/best_model.pkl')
