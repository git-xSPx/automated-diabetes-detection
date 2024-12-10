from config import Config
from TeachModels.logisticRegression import makeLogisticRegression
from TeachModels.XGBoost import makeXGBoost

# Навчаємо модель по алгоритму LogisticRegression 
#makeLogisticRegression(Config)
makeXGBoost(Config)