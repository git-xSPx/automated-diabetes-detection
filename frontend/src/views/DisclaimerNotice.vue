<template>
  <v-container class="fill-height" fluid>
    <!-- Center the card horizontally and vertically -->
    <v-row align="center" justify="center">
      <!-- Card column with responsive width -->
      <v-col cols="12" md="8" lg="6">
        <v-card class="pa-5">
          <!-- Grid container inside the card -->
          <v-container>
            <!-- Title Row -->
            <v-row>
              <v-col cols="12">
                <v-card-title class="text-h5">
                  Увага
                </v-card-title>
              </v-col>
            </v-row>

            <!-- Text Row -->
            <v-row>
              <v-col cols="12">
                <v-card-text>
                  Цей сервіс надається "як є". Використання на власний ризик.
                  Ми не несемо відповідальності за будь-які наслідки.
                </v-card-text>
              </v-col>
            </v-row>

            <!-- Button Row -->
            <v-row justify="center">
              <v-col cols="12" md="4" class="text-center">
                <v-btn color="primary" @click="goToForm">
                  Спробувати безкоштовно
                </v-btn>
              </v-col>
            </v-row>
          </v-container>
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
/* Styling for the card */
.v-card {
  padding: 32px;
}

/* Center and style the title */
.v-card-title {
  font-size: 28px;
  margin-bottom: 16px;
  text-align: center;
}

/* Center and style the text */
.v-card-text {
  font-size: 16px;
  margin-bottom: 24px;
  text-align: center;
}

/* Responsive button width */
.v-btn {
  width: 100%;
}

/* Adjust button width on medium and larger screens */
@media (min-width: 960px) {
  .v-btn {
    width: auto;
  }
}
</style>
