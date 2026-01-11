import os 
import sys 
from dataclasses import dataclass
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor 
from sklearn.linear_model import ElasticNet  
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pk1")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() 

    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, : -1],
                train_array[:, -1],
                test_array[:, : -1],
                test_array[: , -1]
            )

            models = {
                "Elastic Net": ElasticNet(max_iter = 100000),
                "Random Forest": RandomForestRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Adaboost Regressor": AdaBoostRegressor(),
                "XGBoost Regressor": XGBRegressor(random_state = 0, objective = "reg:squarederror", tree_method = "hist", n_jobs = -1),
                "Catboost Regressor": CatBoostRegressor(random_seed = 0, loss_function = 'RMSE', verbose = 0, allow_writing_files = False)
            }

            params = {
                "Elastic Net":{
                'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], 
                'l1_ratio': np.arange(0.05, 1.05, 0.05)},
                
                "Random Forest":{
                "model__n_estimators": [200, 500], 
                "model__max_depth": [None, 5, 10], 
                "model__min_samples_split": [2, 5]},
                
                "K-Neighbors Regressor":{
                "model__n_neighbors": [3, 5, 7, 11], 
                "model__weights": ["uniform", "distance"]},
                
                "Adaboost Regressor":{
                "model__n_estimators": [100, 300, 600], 
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2], 
                "model__loss": ["linear", "square", "exponential"]},

                "XGBoost Regressor":{
                "model__n_estimators": [100, 300, 600],
                "model__max_depth": [3, 5, 8],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__min_child_weight": [1, 5],
                "model__reg_alpha": [0.0, 0.1, 1.0],
                "model__reg_lambda": [1.0, 5.0, 10.0]},

                "Catboost Regressor":{
                "model__iterations": [500, 1000],
                "model__depth": [4, 6, 8, 10],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__l2_leaf_reg": [1, 3, 10],
                "model__subsample": [0.8, 1.0] }
                

            }

            model_report:dict = evaluate_models(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test,
                                               models = models, params = params)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square



        except Exception as e:
            raise CustomException(e, sys)