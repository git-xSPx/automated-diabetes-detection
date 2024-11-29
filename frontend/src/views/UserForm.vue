<template>
  <v-container>
    <v-form ref="form" @submit.prevent="handleSubmit">
      <v-card>
        <v-card-title class="text-h5">
          Введіть свої дані
        </v-card-title>
        <v-card-text>
          <!-- Gender Field -->
          <v-radio-group
            v-model="formData.gender"
            :rules="[rules.required]"
            label="Стать"
          >
            <v-radio label="Чоловіча" value="Male"></v-radio>
            <v-radio label="Жіноча" value="Female"></v-radio>
          </v-radio-group>

          <!-- Age Field -->
          <v-text-field
            v-model="formData.age"
            :rules="[rules.required, rules.age]"
            label="Вік"
            type="number"
            min="1"
            max="120"
          ></v-text-field>

          <!-- Hypertension Field -->
          <v-radio-group
            v-model="formData.hypertension"
            :rules="[rules.required]"
            label="Гіпертонія"
          >
            <v-radio label="Так" value="1"></v-radio>
            <v-radio label="Ні" value="0"></v-radio>
          </v-radio-group>

          <!-- Heart Disease Field -->
          <v-radio-group
            v-model="formData.heart_disease"
            :rules="[rules.required]"
            label="Серцеві захворювання"
          >
            <v-radio label="Так" value="1"></v-radio>
            <v-radio label="Ні" value="0"></v-radio>
          </v-radio-group>

          <!-- Smoking History Field -->
          <v-select
            v-model="formData.smoking_history"
            :items="smokingOptions"
            :rules="[rules.required]"
            label="Історія паління"
          ></v-select>

          <!-- BMI Field -->
          <v-text-field
            v-model="formData.bmi"
            :rules="[rules.required, rules.bmi]"
            label="ІМТ (Індекс маси тіла)"
            type="number"
          ></v-text-field>

          <!-- HbA1c Level Field -->
          <v-text-field
            v-model="formData.HbA1c_level"
            :rules="[rules.required, rules.hba1c]"
            label="Рівень HbA1c (%)"
            type="number"
          ></v-text-field>

          <!-- Blood Glucose Level Field -->
          <v-row>
            <v-col cols="4">
              <v-select
                v-model="bloodGlucoseUnit"
                :items="bloodGlucoseUnits"
                label="Одиниці"
              ></v-select>
            </v-col>
            <v-col cols="8">
              <v-text-field
                v-model="bloodGlucoseValue"
                :rules="[rules.required, rules.bloodGlucose]"
                :label="`Рівень глюкози (${bloodGlucoseUnit})`"
                type="number"
              ></v-text-field>
            </v-col>
          </v-row>

          <!-- Error Messages -->
          <v-alert
            v-if="errorMessage"
            type="error"
            dismissible
            v-model="showError"
          >
            {{ errorMessage }}
          </v-alert>
        </v-card-text>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn color="primary" type="submit">
            Відправити
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-form>
  </v-container>
</template>

<script>
import axios from 'axios';

export default {
  name: 'UserForm',
  data() {
    return {
      formData: {
        gender: '',
        age: null,
        hypertension: '',
        heart_disease: '',
        smoking_history: '',
        bmi: null,
        HbA1c_level: null,
        blood_glucose_level: null, // This will be set after conversion
      },
      bloodGlucoseUnit: 'mg/dL',
      bloodGlucoseUnits: ['mg/dL', 'mmol/L'],
      bloodGlucoseValue: null,
      smokingOptions: ['No Info', 'Never', 'Former', 'Current'],
      errorMessage: '',
      showError: false,
      rules: {
        required: (value) => !!value || 'Обов’язкове поле',
        age: (value) =>
          (value >= 1 && value <= 120) || 'Вік має бути між 1 та 120',
        bmi: (value) =>
          (value >= 10 && value <= 80) || 'ІМТ має бути між 10 та 80',
        hba1c: (value) =>
          (value >= 2 && value <= 20) ||
          'Рівень HbA1c має бути між 2% та 20%',
        bloodGlucose: (value) => {
          if (!value) return 'Обов’язкове поле';
          let mgValue =
            this.bloodGlucoseUnit === 'mmol/L' ? value * 18 : value;
          return (
            (mgValue >= 40 && mgValue <= 400) ||
            'Рівень глюкози має бути між 40 та 400 mg/dL'
          );
        },
      },
    };
  },
  methods: {
    handleSubmit() {
      // Конвертація рівня глюкози в mg/dL
      let bloodGlucoseMg =
        this.bloodGlucoseUnit === 'mmol/L'
          ? this.bloodGlucoseValue * 18
          : this.bloodGlucoseValue;

      // Оновлення даних форми
      this.formData.blood_glucose_level = bloodGlucoseMg;

      // Перевірка форми
      if (this.$refs.form.validate()) {
        // Відправка даних на сервер
        axios
          .post('http://127.0.0.1:5000/predict', this.formData)
          .then((response) => {
            if (response.data.diabetes_prediction === 1) {
              this.$router.push({
                name: 'Result',
                params: {
                  message: `Користувач, ймовірно, має діабет. Імовірність: ${
                    response.data.diabetes_probability * 100
                  }%.`,
                },
              });
            } else {
              this.$router.push({
                name: 'Result',
                params: {
                  message: `У користувача, ймовірно, немає діабету. Імовірність: ${
                    response.data.diabetes_probability * 100
                  }%.`,
                },
              });
            }
          })
          .catch((error) => {
            // Обробка помилок
            this.errorMessage = error.response.data.message || 'Сталася помилка';
            this.showError = true;
          });
      }
    },
  },
};
</script>

<style scoped>
/* Додаткові стилі, якщо необхідно */
</style>
