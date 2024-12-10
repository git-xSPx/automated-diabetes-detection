import pandas as pd
import numpy as np
import os
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def makeCatBoost(Config):
    # 1. Завантаження даних
    data = pd.read_csv(os.path.join(Config.TEACH_DIR, 'diabetes_prediction_dataset.csv'))

    # 2. Попередня обробка даних
    # Перевірка на пропущені значення
    print("Пропущені значення в кожній колонці:")
    print(data.isnull().sum())

    # Вказання категоріальних колонок
    categorical_columns = ['gender', 'smoking_history']

    # Масштабування числових даних
    numerical_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # 3. Розділення даних на ознаки та цільову змінну
    X = data.drop('diabetes', axis=1)
    y = data['diabetes']

    # Розділення на тренувальні та тестові дані
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Створення пула даних для CatBoost
    train_pool = Pool(data=X_train, label=y_train, cat_features=[X.columns.get_loc(col) for col in categorical_columns])
    test_pool = Pool(data=X_test, label=y_test, cat_features=[X.columns.get_loc(col) for col in categorical_columns])

    # 5. Навчання моделі CatBoost
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100
    )
    model.fit(train_pool, eval_set=test_pool)

    # 6. Оцінка моделі
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nКласифікаційний звіт:")
    print(classification_report(y_test, y_pred))

    print("\nМатриця плутанини:")
    print(confusion_matrix(y_test, y_pred))

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC: {roc_auc}")

    # 7. Побудова графіку ROC-кривої
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

    # 8. Збереження моделі для подальшого використання
    from joblib import dump
    dump(model, os.path.join(Config.MODELS_DIR, 'diabetes_catboost_model.joblib'))
    print("\nМодель збережена у файлі 'diabetes_catboost_model.joblib'.")

    dump(scaler, os.path.join(Config.MODELS_DIR, 'diabetes_scaler.joblib'))
    print("Скейлер збережено у файл 'diabetes_scaler.joblib'")
