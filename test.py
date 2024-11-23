import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
from config import Config

# 1. Завантаження моделі
model = load(Config.MODELS_DIR + 'diabetes_logisticRegression_model.joblib')
print("Модель завантажено!")

# 2. Параметри нового пацієнта
# Female,20.0,0,0,never,27.32,6.6,85,0
data = {
    "gender": "Female",  # Female або Male
    "age": 60.0,
    "hypertension": 1,  # 0 = Ні, 1 = Так
    "heart_disease": 0,  # 0 = Ні, 1 = Так
    "smoking_history": "never",  # never, former, current, No Info
    "bmi": 27.32,
    "HbA1c_level": 7.5,
    "blood_glucose_level": 120
}

# 3. Попередня обробка даних пацієнта
label_encoders = {
    "gender": {"Female": 0, "Male": 1},
    "smoking_history": {"No Info": 2, "never": 0, "former": 1, "current": 1, "not current": 1}
}

# Закодування змінних
new_patient_encoded = [
    label_encoders["gender"][data["gender"]],
    data["age"],
    data["hypertension"],
    data["heart_disease"],
    label_encoders["smoking_history"][data["smoking_history"]],
    data["bmi"],
    data["HbA1c_level"],
    data["blood_glucose_level"]
]

# Приведення даних до масиву
new_patient_encoded = np.array(new_patient_encoded).reshape(1, -1)  # Правильна форма (1, 8)
print("Закодовані дані пацієнта:")
print(new_patient_encoded)

# Імена колонок, які використовувалися під час навчання моделі
feature_names = ['gender', 'age', 'hypertension', 'heart_disease', 
                 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Створіть DataFrame для нових даних
new_patient_df = pd.DataFrame(new_patient_encoded, columns=feature_names)  # Використання форми (1, 8)

# Завантаження скейлера
scaler = load(Config.MODELS_DIR + 'diabetes_scaler.joblib')

# Масштабування числових ознак
new_patient_df.loc[:, ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = scaler.transform(
    new_patient_df[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
)

print("Масштабовані дані пацієнта:")
print(new_patient_df)

# 4. Прогноз
diabetes_prediction = model.predict(new_patient_df)
diabetes_probability = model.predict_proba(new_patient_df)[:, 1]

# Виведення результатів
if diabetes_prediction[0] == 1:
    print(f"У пацієнта ймовірно є діабет (ймовірність: {diabetes_probability[0]:.2f}).")
else:
    print(f"У пацієнта ймовірно немає діабету (ймовірність: {diabetes_probability[0]:.2f}).")
