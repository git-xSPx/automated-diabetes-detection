<template>
  <v-container class="fill-height" fluid>
    <v-row align="center" justify="center">
      <v-col cols="12" md="8" lg="6">
        <v-card class="pa-5">
          <v-card-title class="text-h5">
            Результат прогнозу
          </v-card-title>
          <v-card-text>
            {{ message }}
          </v-card-text>
          <v-card-actions>
            <v-btn color="primary" @click="goBack">
              Повернутися
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
export default {
  name: 'PredictionResult',
  props: ['message'],
  methods: {
    goBack() {
      // Повернення до форми
      this.$router.push({ name: 'UserForm' });
    },
  },
};
</script>

<style scoped>
/* Додаткові стилі, якщо необхідно */
</style>
