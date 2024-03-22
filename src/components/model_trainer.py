import os 
import sys

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.metrics import r2_score

from src.exception import CustomException

from src.logger import logging

from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifact','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_training_config=ModelTrainerConfig()
        
    def intiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('split training and test input data')
            
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={'Linear_regression':LinearRegression(),
                    'Decision_tree':DecisionTreeRegressor(),
            }
            
            
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models=models)
            logging.info(f'model report :{model_report}')
            
            best_model_score=max(sorted((model_report.values())))
            
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model=models[best_model_name]
            
            save_object(file_path=self.model_training_config.trained_model_file_path,
                        obj=best_model)
            
            
            
            return model_report
            
        except Exception as e:
            raise CustomException(e,sys)
        