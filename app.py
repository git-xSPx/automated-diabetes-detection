from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import Config

# Ініціалізація Flask-додатку
app = Flask(__name__)

# Маршрут для передбачення
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Завантаження моделі
        model = load(Config.MODELS_DIR + 'diabetes_logisticRegression_model.joblib')

        # 2. Отримання даних із запиту
        data = request.json

        # Валідація вхідних даних
        required_fields = ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Field '{field}' is missing"}), 400

        # 3. Попередня обробка даних пацієнта
        label_encoders = {
            "gender": {"Female": 0, "Male": 1},
            "smoking_history": {"No Info": 2, "never": 0, "former": 1, "current": 1, "not current": 1}
        }

        # Перетворення вхідних даних
        new_patient_encoded = [
            label_encoders["gender"][data["gender"]],
            float(data["age"]),
            int(data["hypertension"]),
            int(data["heart_disease"]),
            label_encoders["smoking_history"][data["smoking_history"]],
            float(data["bmi"]),
            float(data["HbA1c_level"]),
            float(data["blood_glucose_level"])
        ]

        # Приведення даних до масиву
        new_patient_encoded = np.array(new_patient_encoded).reshape(1, -1)  # Правильна форма (1, 8)

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

        # 4. Прогноз
        diabetes_prediction = model.predict(new_patient_df)
        diabetes_probability = model.predict_proba(new_patient_df)[:, 1]

        # Формування відповіді
        result = {
            "diabetes_prediction": int(diabetes_prediction),  # 0 або 1
            "diabetes_probability": round(float(diabetes_probability), 2)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Запуск Flask-додатку
if __name__ == '__main__':
    app.run(debug=True)
