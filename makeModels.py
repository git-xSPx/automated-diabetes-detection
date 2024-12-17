from config import Config
from TeachModels.logisticRegression import makeLogisticRegression
from TeachModels.XGBoost import makeXGBoost
from TeachModels.SVM import makeSVM
from TeachModels.KNN import makeKNN

# Навчаємо модель по алгоритму LogisticRegression
#makeLogisticRegression(Config)
#makeXGBoost(Config)
#makeSVM(Config)
makeKNN(Config, 5)
