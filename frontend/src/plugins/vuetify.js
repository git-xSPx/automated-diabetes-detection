import { createVuetify } from 'vuetify';
import * as components from 'vuetify/components';
import * as directives from 'vuetify/directives';
import 'vuetify/styles'; // Ensure you are importing Vuetify styles
import '@mdi/font/css/materialdesignicons.css' // Ensure you are using css-loader

export default createVuetify({
  components,
  directives,
  theme: {
    themes: {
      light: {
        primary: '#1976D2', // Primary color
        secondary: '#424242',
        accent: '#82B1FF',
        error: '#D32F2F',
        info: '#2196F3',
        success: '#4CAF50',
        warning: '#FFC107',
      },
    },
  },
  icons: {
    defaultSet: 'mdi', // This is already the default value - only for display purposes
  },
});
