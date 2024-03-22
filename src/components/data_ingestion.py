import os 
import sys
from src.exception import CustomException
import pandas as pd
from src.logger import logging 

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass 
class DataIngestionConfig:
    train_data_path=os.path.join('artifact','train.csv')
    test_data_path=os.path.join('artifact','test.csv')
    raw_data_path=os.path.join('artifact','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    
    def intiate_data_ingestion(self):
        logging.info('entered the data ingestion method or component')
        
        try:
            df=pd.read_csv('notebook\data\gemstone.csv')
            df.drop('id',inplace=True,axis=1)
            logging.info('Read the dataset as dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info('Train test split intitated')
            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info('ingestion is completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.intiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.intiate_transformation(train_data,test_data)
    
    modeltrainer=ModelTrainer()
    print(modeltrainer.intiate_model_trainer(train_arr,test_arr))
    
        
        



    