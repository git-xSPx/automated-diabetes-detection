<template>
  <v-container>
    <v-form ref="form" @submit.prevent="handleSubmit">
      <v-card>
        <v-card-title class="text-h5">
          Введіть свої дані
        </v-card-title>
        <v-card-text>
          <!-- Wrap form fields in v-row -->
          <v-row>
            
            <!-- Gender Field -->
            <v-col cols="12" md="4">
              <v-radio-group
                v-model="formData.gender"
                :rules="[rules.required]"
                label="Стать"
                color="primary"
              >
                <v-radio label="Чоловіча" value="Male"></v-radio>
                <v-radio label="Жіноча" value="Female"></v-radio>
              </v-radio-group>
            </v-col>

            <!-- Hypertension Field -->
            <v-col cols="12" md="4">
              <v-radio-group
                v-model="formData.hypertension"
                :rules="[rules.required]"
                label="Гіпертонія"
                color="primary"
              >
                <v-radio label="Так" value="1"></v-radio>
                <v-radio label="Ні" value="0"></v-radio>
              </v-radio-group>
            </v-col>

            <!-- Heart Disease Field -->
            <v-col cols="12" md="4">
              <v-radio-group
                v-model="formData.heart_disease"
                :rules="[rules.required]"
                label="Серцеві захворювання"
                color="primary"
              >
                <v-radio label="Так" value="1"></v-radio>
                <v-radio label="Ні" value="0"></v-radio>
              </v-radio-group>
            </v-col>

            <!-- Age Field -->
            <v-col cols="12" md="6">
              <v-text-field
              v-model="formData.age"
              :rules="[rules.required, rules.age]"
              label="Вік"
              type="number"
              min="1"
              max="120"
              ></v-text-field>
            </v-col>

            <!-- Smoking History Field -->
            <v-col cols="12" md="6">
              <v-select
                v-model="formData.smoking_history"
                :items="smokingOptions"
                :rules="[rules.required]"
                label="Історія паління"
              ></v-select>
            </v-col>

            <!-- BMI Field -->
            <v-col cols="12" md="6">
              <v-text-field
                v-model="formData.bmi"
                :rules="[rules.required, rules.bmi]"
                label="ІМТ (Індекс маси тіла)"
                type="number"
              ></v-text-field>
            </v-col>

            <!-- HbA1c Level Field -->
            <v-col cols="12" md="6">
              <v-text-field
                v-model="formData.HbA1c_level"
                :rules="[rules.required, rules.hba1c]"
                label="Рівень HbA1c (%)"
                type="number"
              ></v-text-field>
            </v-col>

            <!-- Blood Glucose Level Field -->
            <v-col cols="12" md="6">
              <v-row>
                <v-col cols="5">
                  <v-select
                    v-model="bloodGlucoseUnit"
                    :items="bloodGlucoseUnits"
                    label="Одиниці"
                  ></v-select>
                </v-col>
                <v-col cols="7">
                  <v-text-field
                    v-model="bloodGlucoseValue"
                    :rules="[rules.required, rules.bloodGlucose]"
                    :label="`Рівень глюкози (${bloodGlucoseUnit})`"
                    type="number"
                  ></v-text-field>
                </v-col>
              </v-row>
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

        <!-- Form Actions -->
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
          (value >= 2 && value <= 20) || 'Рівень HbA1c має бути між 2% та 20%',
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
        axios({
              method: 'POST',
              url: '/predict',
              data: this.formData,
              headers: {
                'Access-Control-Allow-Origin': '*',
                'Content-type': 'application/json',
              }
          })
//          .post('http://127.0.0.1:5000/predict', this.formData)
          .then((response) => {
            const probability = response.data.diabetes_probability * 100;
            if (response.data.diabetes_prediction === 1) {
              this.$router.push({
                name: 'Result',
                query: {
                  message: `Користувач, ймовірно, має діабет. Імовірність: ${probability}%.`,
                },
              });
            } else {
              this.$router.push({
                name: 'Result',
                query: {
                  message: `У користувача, ймовірно, немає діабету. Імовірність: ${probability}%.`,
                },
              });
            }
          })
          .catch((error) => {
            // Обробка помилок
            this.errorMessage =
              error.response?.data?.message || 'Сталася помилка';
            this.showError = true;
          });
      }
    },
  },
};
</script>

<style scoped>
.v-card {
  padding: 24px;
}

.v-card-title {
  margin-bottom: 16px;
}

.v-card-text {
  margin-bottom: 16px;
}

.v-text-field,
.v-select,
.v-radio-group {
  margin-bottom: 24px;
}

.v-radio-group .v-radio {
  margin-bottom: 8px;
}

.v-btn {
  margin-top: 16px;
}

.v-alert {
  margin-top: 16px;
}

.v-tooltip {
  font-size: 12px;
}
</style>
