<template>
  <v-app>
    <!-- Верхня панель додатку -->
    <v-app-bar app color="primary" dark>
      <v-container>
        <v-row align="center">
          <!-- Колонка з логотипом та заголовком -->
          <v-col cols="12" md="3" class="d-flex align-center">
            <!-- Логотип -->
            <v-img
              src="@/assets/ADD_logo.png"
              alt="Логотип"
              max-height="40"
              contain
              class="mr-2"
            ></v-img>
            <!-- Заголовок -->
            <v-toolbar-title class="white--text">
              Прогноз діабету
            </v-toolbar-title>
          </v-col>
          <!-- Колонка з навігаційними кнопками -->
          <v-col cols="12" md="9" class="text-right">
            <v-btn text to="/" class="white--text">Головна</v-btn>
            <v-btn text to="/form" class="white--text">Форма</v-btn>
            <v-btn text to="/about" class="white--text">Про нас</v-btn>
          </v-col>
        </v-row>
      </v-container>
    </v-app-bar>

    <!-- Основний контент -->
    <v-main>
      <v-container>
        <!-- Роутер-в'ю для відображення компонентів -->
        <router-view></router-view>
      </v-container>
    </v-main>

    <!-- Нижній колонтитул -->
    <v-footer app color="primary" dark>
      <v-container>
        <v-row justify="center">
          <v-col cols="12" md="6" class="text-center">
            © {{ new Date().getFullYear() }} Прогноз діабету
          </v-col>
        </v-row>
      </v-container>
    </v-footer>
  </v-app>
</template>

<script>
export default {
  name: 'App',
};
</script>

<style>
/* Стилі для заголовка в верхній панелі */
.v-toolbar-title {
  font-size: 24px;
  font-weight: 500;
}

/* Стилі для навігаційних кнопок в верхній панелі */
.v-app-bar .v-btn {
  margin-left: 16px;
}

/* Стилі для основного контейнера */
.v-main {
  background-color: #f5f5f5;
}

/* Стилі для контейнера контенту */
.v-container {
  padding-top: 24px;
  padding-bottom: 24px;
}

/* Стилі для нижнього колонтитулу */
.v-footer {
  padding-top: 16px;
  padding-bottom: 16px;
}
</style>
