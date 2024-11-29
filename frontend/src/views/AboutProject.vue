<template>
  <v-container>
    <v-card>
      <!-- Початок сітки всередині картки -->
      <v-container>
        <!-- Рядок з логотипом та заголовком -->
        <v-row align="center" justify="space-between">
          <v-col cols="12" md="8">
            <v-card-title class="text-h5">
              Автоматизована система прогнозування виявлення цукрового діабету
            </v-card-title>
            <v-text>
              <p>Проект було виконано в рамках виконання дипломної роботи студента групи АТ-28м <strong>Підопригори С.В.</strong></p>
              <p>Науковий керівник: кандидат технічних наук <strong>Ткачук А.Г.</strong>.</p>
              <p>Тема дипломної роботи: <strong><em>"Автоматизована система прогнозування виявлення цукрового діабету на основі медичних даних пацієнтів".</em></strong></p>
            </v-text>
          </v-col>
          <v-col cols="12" md="4" class="text-right">
            <!-- Логотип -->
            <v-img
              src="@/assets/ADD_logo.png"
              alt="Логотип"
              max-height="140"
              contain
            ></v-img>
          </v-col>
        </v-row>

        <!-- Рядок з контентом -->
        <v-row>
          <v-col cols="12">
            <!-- Використовуємо компоненти типографії для оформлення тексту -->
            <v-card-text>
              <v-divider></v-divider>
              <v-card-subtitle class="mt-4">
                Про проект
              </v-card-subtitle>
              <v-text>
                Проект "Автоматизована система прогнозування виявлення цукрового діабету" — це інноваційне рішення, яке допомагає автоматично оцінювати ймовірність наявності цукрового діабету у пацієнтів на основі їхніх медичних показників. Система створена для підтримки лікарів у прийнятті клінічних рішень, а також для підвищення обізнаності пацієнтів щодо стану їхнього здоров'я.
              </v-text>

              <v-card-subtitle class="mt-4">
                Основна мета проекту
              </v-card-subtitle>
              <v-text>
                Основна мета проекту — створити простий і точний інструмент, який допоможе швидко аналізувати медичні дані пацієнтів та передбачати ризик діабету. Це дозволяє лікарям:
              </v-text>
              <v-list dense>
                <v-list-item>
                  <v-list-item-content>Прискорити процес первинної діагностики.</v-list-item-content>
                </v-list-item>
                <v-list-item>
                  <v-list-item-content>Запропонувати превентивні заходи для груп ризику.</v-list-item-content>
                </v-list-item>
                <v-list-item>
                  <v-list-item-content>Покращити якість лікування, завдяки своєчасній інформації.</v-list-item-content>
                </v-list-item>
              </v-list>

              <v-card-subtitle class="mt-4">
                Як це працює?
              </v-card-subtitle>
              <v-text>
                Система використовує модель машинного навчання, яка була натренована на реальних даних пацієнтів. Основні етапи роботи включають:
              </v-text>
              <v-list dense>
                <v-list-item>
                  <v-list-item-content>
                    <strong>Збір даних:</strong> Пацієнт надає базову інформацію, таку як вік, стать, індекс маси тіла (BMI), рівень глюкози в крові тощо.
                  </v-list-item-content>
                </v-list-item>
                <v-list-item>
                  <v-list-item-content>
                    <strong>Обробка даних:</strong> Дані стандартизуються та проходять попередню обробку для підготовки до аналізу.
                  </v-list-item-content>
                </v-list-item>
                <v-list-item>
                  <v-list-item-content>
                    <strong>Аналіз:</strong> Модель прогнозує ймовірність наявності цукрового діабету на основі вхідних даних.
                  </v-list-item-content>
                </v-list-item>
                <v-list-item>
                  <v-list-item-content>
                    <strong>Результат:</strong> Користувач отримує зрозумілий результат з інформацією про ризик діабету та рекомендаціями.
                  </v-list-item-content>
                </v-list-item>
              </v-list>

              <v-card-subtitle class="mt-4">
                Ключові особливості проекту
              </v-card-subtitle>
              <v-list dense>
                <v-list-item>
                  <v-list-item-content><strong>Точність:</strong> Використання сучасних алгоритмів машинного навчання забезпечує високу точність передбачення.</v-list-item-content>
                </v-list-item>
                <v-list-item>
                  <v-list-item-content><strong>Дружній інтерфейс:</strong> Система інтуїтивно зрозуміла як для лікарів, так і для пацієнтів.</v-list-item-content>
                </v-list-item>
                <v-list-item>
                  <v-list-item-content><strong>Швидкість:</strong> Результати прогнозу доступні в реальному часі.</v-list-item-content>
                </v-list-item>
                <v-list-item>
                  <v-list-item-content><strong>Простота інтеграції:</strong> Може бути інтегрована в медичні інформаційні системи або використовуватись як окремий інструмент.</v-list-item-content>
                </v-list-item>
              </v-list>

              <v-card-subtitle class="mt-4">
                Технології, що використовуються
              </v-card-subtitle>
              <v-text>
                <strong>Моделі машинного навчання:</strong> Logistic Regression (логістична регресія).<br>
                <strong>Бекенд:</strong> Python (Flask), API для інтеграції.<br>
                <strong>Фронтенд:</strong> Простий та інтуїтивний веб-інтерфейс.<br>
                <strong>Інструменти для роботи з даними:</strong> Pandas, Numpy, Scikit-learn.<br>
                <strong>Зберігання моделей:</strong> Joblib для серіалізації моделей та скейлерів.
              </v-text>

              <v-card-subtitle class="mt-4">
                Для кого призначений проект?
              </v-card-subtitle>
              <v-list dense>
                <v-list-item>
                  <v-list-item-content><strong>Лікарі:</strong> Система допомагає швидко оцінювати ризики, що є корисним під час первинного обстеження.</v-list-item-content>
                </v-list-item>
                <v-list-item>
                  <v-list-item-content><strong>Пацієнти:</strong> Дозволяє людям отримати попередню оцінку свого стану та вчасно звернутися до лікаря.</v-list-item-content>
                </v-list-item>
                <v-list-item>
                  <v-list-item-content><strong>Медичні заклади:</strong> Інтеграція з існуючими системами для автоматизації аналізу даних пацієнтів.</v-list-item-content>
                </v-list-item>
              </v-list>

              <v-card-subtitle class="mt-4">
                Як користуватись?
              </v-card-subtitle>
              <v-list dense>
                <v-list-item>
                  <v-list-item-content>Введіть свої дані у форму (вік, стать, історія куріння, рівень глюкози тощо).</v-list-item-content>
                </v-list-item>
                <v-list-item>
                  <v-list-item-content>Надішліть дані для аналізу.</v-list-item-content>
                </v-list-item>
                <v-list-item>
                  <v-list-item-content>Отримайте результати з детальною інформацією про ризик наявності діабету.</v-list-item-content>
                </v-list-item>
              </v-list>

              <v-card-subtitle class="mt-4">
                Наш внесок
              </v-card-subtitle>
              <v-text>
                Проект "Система передбачення діабету" є кроком до покращення якості медичної діагностики. Ми прагнемо зробити медичну допомогу більш доступною, точною та інноваційною.
              </v-text>

              <v-card-subtitle class="mt-4">
                Зв'яжіться з нами
              </v-card-subtitle>
              <v-text>
                Якщо у вас є запитання або пропозиції щодо покращення проекту, напишіть нам email: <a href="mailto:at28m_psv@student.ztu.edu.ua">at28m_psv@student.ztu.edu.ua</a>.
              </v-text>
            </v-card-text>
          </v-col>
        </v-row>
      </v-container>
      <!-- Кінець сітки всередині картки -->
    </v-card>
  </v-container>
</template>

<script>
export default {
  name: 'AboutProject',
};
</script>

<style scoped>
/* Стилі для картки */
.v-card {
  padding: 24px;
}

/* Стилі для заголовка */
.v-card-title {
  margin-bottom: 16px;
  font-size: 24px;
  font-weight: bold;
}

/* Стилі для підзаголовків */
.v-card-subtitle {
  font-size: 20px;
  font-weight: bold;
  margin-bottom: 8px;
}

/* Стилі для тексту */
.v-text {
  font-size: 16px;
  line-height: 1.6;
  margin-bottom: 16px;
  display: block;
}

/* Стилі для списків */
.v-list-item-content {
  font-size: 16px;
}

/* Стилі для логотипу */
.v-img {
  max-height: 60px;
}

/* Відступи для списків */
.v-list {
  margin-bottom: 16px;
}
</style>
