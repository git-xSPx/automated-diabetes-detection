<template>
  <v-container class="fill-height" fluid>
    <!-- Центруємо картку по вертикалі та горизонталі -->
    <v-row align="center" justify="center">
      <v-col cols="12" md="8" lg="6">
        <v-card class="pa-5">
          <!-- Початок сітки всередині картки -->
          <v-container>
            <!-- Рядок з заголовком -->
            <v-row>
              <v-col cols="12">
                <v-card-title class="text-h5">
                  Результат прогнозу
                </v-card-title>
              </v-col>
            </v-row>

            <!-- Рядок з повідомленням -->
            <v-row>
              <v-col cols="12">
                <v-card-text>
                  {{ message }}
                </v-card-text>
              </v-col>
            </v-row>

            <!-- Рядок з кнопкою -->
            <v-row justify="center">
              <v-col cols="12" md="4" class="text-center">
                <v-btn color="primary" @click="goBack">
                  Повернутися
                </v-btn>
              </v-col>
            </v-row>
          </v-container>
          <!-- Кінець сітки всередині картки -->
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
export default {
  name: 'PredictionResult',
  computed: {
    message() {
      return this.$route.query.message || 'Немає результату';
    },
  },
  methods: {
    goBack() {
      // Повернення до форми
      this.$router.push({ name: 'UserForm' });
    },
  },
};
</script>

<style scoped>
/* Стилі для картки */
.v-card {
  padding: 32px;
}

/* Центруємо та стилізуємо заголовок */
.v-card-title {
  font-size: 28px;
  margin-bottom: 16px;
  text-align: center;
}

/* Центруємо та стилізуємо текст повідомлення */
.v-card-text {
  font-size: 18px;
  margin-bottom: 24px;
  text-align: center;
}

/* Розмір кнопки */
.v-btn {
  width: 100%;
}

/* Налаштування ширини кнопки на великих екранах */
@media (min-width: 960px) {
  .v-btn {
    width: auto;
  }
}
</style>
