import os
import sys
import dill

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