import os
import sys
import dill

from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    """
    try:
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logging.error(f"Error occurred while saving object at {file_path}")
        raise CustomException(f"Error occurred while saving object at {file_path}", sys) from e
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)