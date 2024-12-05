const express = require('express');
const path = require('path');
const app = express();

// Обслуговування статичних файлів з папки dist
app.use(express.static(path.join(__dirname, 'dist')));

// Обробка всіх GET-запитів, повертаючи index.html
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

// Встановлення порту
const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`Сервер запущено на порту ${PORT}`);
});
