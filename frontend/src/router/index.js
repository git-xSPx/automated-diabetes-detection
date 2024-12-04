import { createRouter, createWebHistory } from 'vue-router';
import DisclaimerNotice from '../views/DisclaimerNotice.vue';
import UserForm from '../views/UserForm.vue';
import PredictionResult from '../views/PredictionResult.vue';
import AboutProject from '../views/AboutProject.vue';

const routes = [
  {
    path: '/',
    name: 'Disclaimer',
    component: DisclaimerNotice,
  },
  {
    path: '/form',
    name: 'UserForm',
    component: UserForm,
  },
  {
    path: '/result',
    name: 'Result',
    component: PredictionResult,
    props: true,
  },
  {
    path: '/about',
    name: 'About',
    component: AboutProject,
  },
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes,
});

export default router;
