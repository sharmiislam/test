# my_package/data_preprocessing.py

import pandas as pd
import logging

def load_data(eye_data_path, demo_data_path, columns_to_load, nrows=None):
    logging.info("Loading Eye Data")
    eye_data = pd.read_csv(eye_data_path, usecols=columns_to_load, nrows=nrows)
    
    logging.info("Loading Demographic Data")
    demo_data = pd.read_csv(demo_data_path, usecols=columns_to_load, nrows=nrows)
    
    logging.info("Data loaded successfully")
    
    return eye_data, demo_data

def get_common_columns(eye_data, demo_data):
    try:
        common_columns = eye_data.columns.intersection(demo_data.columns).tolist()
        logging.info(f"Common Columns: {common_columns}")
        return common_columns, common_columns  # Adjust based on actual need
    except Exception as e:
        logging.error(f"Error finding common columns: {e}")
        raise

