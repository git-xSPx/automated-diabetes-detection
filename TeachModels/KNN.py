# Імпортуємо необхідні бібліотеки
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from joblib import dump

def makeKNN(Config, k=5):
    # 1. Завантаження даних
    data = pd.read_csv(os.path.join(Config.TEACH_DIR, 'diabetes_prediction_dataset.csv'))

    # 2. Попередня обробка даних
    print("Пропущені значення в кожній колонці:")
    print(data.isnull().sum())

    # Кодіфікація категоріальних даних
    categorical_columns = ['gender', 'smoking_history']
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Масштабування числових даних
    scaler = StandardScaler()
    numerical_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # 3. Розділення даних на ознаки та цільову змінну
    X = data.drop('diabetes', axis=1)
    y = data['diabetes']

    # Розділення на тренувальні та тестові дані
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Навчання моделі K-NN
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # 5. Оцінка моделі
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nКласифікаційний звіт:")
    print(classification_report(y_test, y_pred))

    print("\nМатриця плутанини:")
    print(confusion_matrix(y_test, y_pred))

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC: {roc_auc}")

    # 6. Побудова графіку ROC-кривої
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

    # 7. Збереження моделі та скейлера
    dump(model, os.path.join(Config.MODELS_DIR, 'diabetes_knn_model.joblib'))
    print("\nМодель збережена у файлі 'diabetes_knn_model.joblib'.")

    dump(scaler, os.path.join(Config.MODELS_DIR, 'diabetes_knn_scaler.joblib'))
    print("Скейлер збережено у файл 'diabetes_knn_scaler.joblib'")