from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def svm_classifier(kernel: str = "linear", C: float = 1.0, degree: int = 3, gamma: str = "scale"):
   """
   TODO: Return a scikit-learn SVC model with the specified parameters.
   """
   return SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)




def svm_regressor(kernel: str = "linear", C: float = 1.0, degree: int = 3, gamma: str = "scale"):
   """
   TODO: Return a scikit-learn SVR model with the specified parameters.
   """
   return SVR(kernel=kernel, C=C, degree=degree, gamma=gamma)


def evaluate_classifier(model, X_test, y_test):
   """
   TODO: Compute and return accuracy, precision, recall, and F1 score
   """
   y_pred = model.predict(X_test, y_test)
   acc = accuracy_score(y_test, y_pred)
   prec = precision_score(y_test, y_pred)
   rec = recall_score(y_test, y_pred)
   f1 = f1_score(y_test, y_pred)
  
   return acc, prec, rec, f1




def evaluate_regressor(model, X_test, y_test):
   """
   TODO: Compute and return MAE, RMSE, and R2
   """
   y_pred = model.predict(X_test, y_test)
   mae = mean_absolute_error(y_test, y_pred)
   rse = np.sqrt(mean_squared_error(y_test, y_pred))
   r2 = r2_score(y_test, y_pred)


   return mae, rse, r2



