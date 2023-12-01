import os
from src.gemstone import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.gemstone.exception import CustomException
from src.gemstone.logger import logging
import pickle
import os
import sys

from src.gemstone.utils.common import save_object
from src.gemstone.entity.config_entity import DataTransformationConfig



class DataTransformation:
    # Define categorical_columns as a class-level attribute
    categorical_columns = ['cut', 'color', 'clarity']

    def __init__(self, config):
        self.config = config

    def get_data_transformation_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            logging.info(f'Categorical Columns : {categorical_cols}')
            logging.info(f'Numerical Columns   : {numerical_cols}')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_cols),
                    ('cat_pipeline', cat_pipeline, categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.info('Exception occurred in Data Transformation Phase')
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        try:
            # Corrected path construction
            data_path = os.path.join("artifacts", "data_ingestion", "gemstone.csv")

            # Reading data
            data = pd.read_csv(data_path)

            # Split the data into training and test sets. (0.75, 0.25) split.
            train, test = train_test_split(data, test_size=0.25, random_state=42)

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']

            input_feature_train_df = train.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train[target_column_name]

            input_feature_test_df = test.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Convert NumPy arrays to DataFrames
            train_df = pd.DataFrame(train_arr, columns=list(input_feature_train_df.columns) + [target_column_name])
            test_df = pd.DataFrame(test_arr, columns=list(input_feature_test_df.columns) + [target_column_name])

            # Save train and test data to CSV
            train_df.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
            test_df.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)


            # Save preprocessing object
            preprocessing_obj_file = os.path.join("artifacts", 'data_transformation', 'preprocessing_obj.pkl')
            with open(preprocessing_obj_file, 'wb') as file:
                pickle.dump(preprocessing_obj, file)

            logging.info("Saved preprocessing object.")
            logging.info("Transformation of the data is completed")

            return train_arr, test_arr, preprocessing_obj_file

        except Exception as e:
            logging.info('Exception occurred in initiate_data_transformation function')
            raise CustomException(e, sys)
     
