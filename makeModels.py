from config import Config
from TeachModels.logisticRegression import makeLogisticRegression
from TeachModels.XGBoost import makeXGBoost
from TeachModels.SVM import makeSVM

# Навчаємо модель по алгоритму LogisticRegression
#makeLogisticRegression(Config)
#makeXGBoost(Config)
makeSVM(Config)