<template>
  <v-app>
    <v-app-bar app color="primary" dark>
      <v-toolbar-title>Прогноз діабету</v-toolbar-title>
      <v-spacer></v-spacer>
      <v-btn text to="/">Головна</v-btn>
      <v-btn text to="/form">Форма</v-btn>
      <v-btn text to="/about">Про нас</v-btn>
    </v-app-bar>
    <v-main>
      <router-view></router-view>
    </v-main>
    <v-footer app color="primary" dark>
      <v-col class="text-center">
        © {{ new Date().getFullYear() }} Прогноз діабету
      </v-col>
    </v-footer>
  </v-app>
</template>

<script>
export default {
  name: 'App',
};
</script>

<style>
/* Додаткові стилі, якщо необхідно */
</style>
