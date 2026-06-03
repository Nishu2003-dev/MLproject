import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException 
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path :str =os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "KNN":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor(),
                "Gradient Boosting Regressor":GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression()
            }
            params = {
                   "Decision Tree": {
                   'criterion': ['squared_error', 'friedman_mse'],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5]
                     },
                    "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10],
                    'min_samples_split': [2, 5]
                     },
                    "KNN": {                                  
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                        },
                    "XGBRegressor": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5]
                     },
                    "CatBoosting Regressor": {
                     'iterations': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'depth': [4, 6]
                        },
                    "AdaBoost Regressor": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 1.0]
                       },
                    "Gradient Boosting Regressor": {
                       'n_estimators': [50, 100],
                     'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                        },
                    "Linear Regression": {}
                       }
            
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                              models=models,param=params )
            
            #To get the best model score from report dict
            best_model_score=max(sorted(model_report.values()))
            
            #best model name from report dict
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found",sys)
            
            logging.info(f"Best model found on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted =best_model.predict(X_test)
            
            r2 =r2_score(y_test,predicted)
            return r2
        
        except Exception as e:
            raise CustomException(e,sys)
            
