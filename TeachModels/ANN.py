# Імпортуємо необхідні бібліотеки
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import Adam
from joblib import dump

def makeANN(Config):
    # 1. Завантаження даних
    data = pd.read_csv(os.path.join(Config.TEACH_DIR, 'diabetes_prediction_dataset.csv'))

    # 2. Попередня обробка даних
    print("Пропущені значення в кожній колонці:")
    print(data.isnull().sum())

    # Кодіфікація категоріальних даних
    categorical_columns = ['gender', 'smoking_history']
    label_encoders = {
        "gender": {"Female": 0, "Male": 1},
        "smoking_history": {"No Info": 2, "never": 0, "ever": 0, "former": 1, "current": 1, "not current": 1}
    }

    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Масштабування числових даних
    scaler = StandardScaler()
    numerical_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # 3. Розділення даних на ознаки та цільову змінну
    X = data.drop('diabetes', axis=1)
    y = data['diabetes']

    # Розділення на тренувальні та тестові дані
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Створення та навчання ANN
    # Створення моделі
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))  # Вхідний шар + перший прихований шар
    model.add(Dense(16, activation='relu'))  # Другий прихований шар
    model.add(Dense(1, activation='sigmoid'))  # Вихідний шар (для класифікації)

    # Компіляція моделі
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Навчання моделі
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1)

    # 5. Оцінка моделі
    y_proba = model.predict(X_test).flatten()
    y_pred = (y_proba > 0.5).astype(int)

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
    model.save(os.path.join(Config.MODELS_DIR, 'diabetes_ann_model.h5'))
    print("\nМодель збережена у файлі 'diabetes_ann_model.h5'.")

    dump(scaler, os.path.join(Config.MODELS_DIR, 'diabetes_scaler.joblib'))
    print("Скейлер збережено у файл 'diabetes_scaler.joblib'")
