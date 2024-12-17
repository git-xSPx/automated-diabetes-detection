from flask import Flask, request, jsonify
from joblib import load
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import Config
from flask_cors import CORS

# Завантаження моделей
modelLR = load(os.path.join(Config.MODELS_DIR, 'diabetes_logisticRegression_model.joblib'))
modelXGBoost = load(os.path.join(Config.MODELS_DIR, 'diabetes_xgboost_model.joblib'))
# Завантаження скейлерів
scalerLR = load(os.path.join(Config.MODELS_DIR, 'diabetes_logisticRegression_scaler.joblib'))
scalerXGBoost = load(os.path.join(Config.MODELS_DIR, 'diabetes_xgboost_scaler.joblib'))

# Ініціалізація Flask-додатку
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Маршрут для передбачення
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Отримання даних із запиту
        data = request.json

        # Ініціалізація результату значенням по замовчуванню
        result = {
            "diabetes_prediction_model": "",
            "diabetes_prediction": 0,
            "diabetes_probability": 0.5,
            "msg": "Наші моделі видали суперечливу інформацію, проконсультуйтесь з лікарем!"
        }

        # Отримання результатів передбачення від моделей
        xgBoostResult = predictByModel(data, modelXGBoost, scalerXGBoost, {"diabetes_prediction_model": "XGBoost"})
        lrResult = predictByModel(data, modelLR, scalerLR, {"diabetes_prediction_model": "LR"})

        # Обираємо ту модель результат якої знаходиться "далі" від невизначеності (0.5)
        if abs(0.5 - xgBoostResult["diabetes_probability"]) >= abs(0.5 - lrResult["diabetes_probability"]):
            result = xgBoostResult
        else:
            result = lrResult        

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def predictByModel(data, model, scaler, addData = dict()):
    # Попередня обробка даних пацієнта
    label_encoders = {
        "gender": {"Female": 0, "Male": 1},
        "smoking_history": {"no info": 2, "never": 0, "former": 1, "current": 1, "not current": 1}
    }

    # Перетворення вхідних даних
    new_patient_encoded = [
        label_encoders["gender"][data["gender"]],
        float(data["age"]),
        int(data["hypertension"]),
        int(data["heart_disease"]),
        label_encoders["smoking_history"][data["smoking_history"].lower()],
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

    # Масштабування числових ознак
    new_patient_df.loc[:, ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = scaler.transform(
        new_patient_df[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
    )

    # Прогноз
    diabetes_prediction = model.predict(new_patient_df)
    diabetes_probability = model.predict_proba(new_patient_df)[:, 1]

    # Формування відповіді
    result = {
        "diabetes_prediction_model": "",
        "diabetes_prediction": int(diabetes_prediction[0]),  # 0 або 1
        "diabetes_probability": round(float(diabetes_probability[0]), 2),
        "msg": ""
    }

    # Якщо є додаткові дані, добавляємо/поновлюємо їх
    if len(addData) > 0:
        result.update(addData)

    return result

# Запуск Flask-додатку
if __name__ == '__main__':
    #app.run(debug=True)
    app.run()
