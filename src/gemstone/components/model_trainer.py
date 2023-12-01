import pandas as pd
import os
from src.gemstone import logger
import pickle
# Basic Import
import numpy as np
import pandas as pd

# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.gemstone.exception import CustomException
from src.gemstone.logger import logging
from src.gemstone.utils.common import save_object
from src.gemstone.utils.common import evaluate_models
from src.gemstone.utils.common import print_evaluated_results
from src.gemstone.utils.common import model_metrics
from sklearn.metrics import r2_score
import dill
from src.gemstone.entity.config_entity import ModelTrainerConfig

from dataclasses import dataclass
import sys
import os



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config  

    @staticmethod
    def save_object(file_path, obj):
        with open(file_path, "wb") as file:
            pickle.dump(obj, file) 
    
    def initiate_model_training(self):
        try:
            # Load train and test data
            train_path = os.path.join("artifacts", "data_transformation", "train.csv")
            test_path = os.path.join("artifacts", "data_transformation", "test.csv")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Splitting Dependent and Independent variables from train and test data')
            xtrain, ytrain, xtest, ytest = (
                train_df.iloc[:, :-1].values,
                train_df.iloc[:, -1].values,
                test_df.iloc[:, :-1].values,
                test_df.iloc[:, -1].values
            )
            
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "GradientBoosting Regressor": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Make sure to define or import the missing functions
            # e.g., evaluate_models, print_evaluated_results, model_metrics
            
            model_report: dict = evaluate_models(xtrain, ytrain, xtest, ytest, models)

            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            # To get the best model score from the dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info('Best model has an R2 score less than 60%')
                raise CustomException('No Best Model Found')
            
            # Save model object
            trained_model_file_path = os.path.join("artifacts", "model_trainer", "model.pkl")
            with open(trained_model_file_path, "wb") as file:
                pickle.dump(best_model, file)
            
            print(f'Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}')
            logging.info('Hyperparameter tuning started for Catboost')

            # Hyperparameter tuning on Catboost
            # Initializing Catboost
            cbr = CatBoostRegressor(verbose=False)

            # Creating the hyperparameter grid
            param_dist = {'depth': [4, 5, 6, 7, 8, 9, 10],
                          'learning_rate': [0.01, 0.02, 0.03, 0.04],
                          'iterations': [300, 400, 500, 600]}

            # Instantiate RandomSearchCV object
            rscv = RandomizedSearchCV(cbr, param_dist, scoring='r2', cv=5, n_jobs=-1)

            # Fit the model
            rscv.fit(xtrain, ytrain)

            # Print the tuned parameters and score
            print(f'Best Catboost parameters: {rscv.best_params_}')
            print(f'Best Catboost Score: {rscv.best_score_}')
            print('\n====================================================================================\n')

            best_cbr = rscv.best_estimator_

            logging.info('Hyperparameter tuning complete for Catboost')

            logging.info('Hyperparameter tuning started for KNN')

            # Initialize KNN
            knn = KNeighborsRegressor()

            # parameters
            k_range = list(range(2, 31))
            param_grid = dict(n_neighbors=k_range)

            # Fitting the cvmodel
            grid = GridSearchCV(knn, param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid.fit(xtrain, ytrain)

            # Print the tuned parameters and score
            print(f'Best KNN Parameters: {grid.best_params_}')
            print(f'Best KNN Score: {grid.best_score_}')
            print('\n====================================================================================\n')

            best_knn = grid.best_estimator_

            logging.info('Hyperparameter tuning Complete for KNN')

            logging.info('Voting Regressor model training started')

            # Creating the final Voting regressor
            er = VotingRegressor([('cbr', best_cbr), ('xgb', XGBRegressor()), ('knn', best_knn)], weights=[3, 2, 1])
            er.fit(xtrain, ytrain)
            print('Final Model Evaluation :\n')
            print_evaluated_results(xtrain, ytrain, xtest, ytest, er)
            logging.info('Voting Regressor Training Completed')

            # Make sure to define or import the missing function
            # e.g., save_object
            
            logging.info('Model pickle file saved')
            # Evaluating Ensemble Regressor (Voting Classifier on test data)
            ytest_pred = er.predict(xtest)

            mae, rmse, r2 = model_metrics(ytest, ytest_pred)
            logging.info(f'Test MAE: {mae}')
            logging.info(f'Test RMSE: {rmse}')
            logging.info(f'Test R2 Score: {r2}')
            logging.info('Final Model Training Completed')
            
            return mae, rmse, r2 
        
        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise e
