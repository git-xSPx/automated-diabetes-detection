<template>
  <v-container class="fill-height" fluid>
    <v-row align="center" justify="center">
      <v-col cols="12" md="8" lg="6">
        <v-card class="pa-5">
          <v-card-title class="text-h5">
            Увага
          </v-card-title>
          <v-card-text>
            Цей сервіс надається "як є". Використання на власний ризик. Ми не несемо відповідальності за будь-які наслідки.
          </v-card-text>
          <v-card-actions>
            <v-spacer></v-spacer>
            <v-btn color="primary" @click="goToForm">
              Спробувати безкоштовно
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
export default {
  name: 'DisclaimerNotice',
  methods: {
    goToForm() {
      // Переходимо до форми вводу даних
      this.$router.push({ name: 'UserForm' });
    },
  },
};
</script>

<style scoped>
/* Додаткові стилі, якщо необхідно */
</style>
