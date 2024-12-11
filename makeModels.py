from config import Config
from TeachModels.logisticRegression import makeLogisticRegression
from TeachModels.XGBoost import makeXGBoost
from TeachModels.ANN import makeANN

# Навчаємо модель по алгоритму LogisticRegression 
#makeLogisticRegression(Config)
#makeXGBoost(Config)
makeANN(Config)