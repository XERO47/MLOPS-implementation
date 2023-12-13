from abc import ABC, abstractmethod
import logging
from sklearn.linear_model import LinearRegression   
class Model(ABC):
    @abstractmethod
    def train(self,x_train,y_train):
        pass

class LinearRegressionModel(Model):
    def __init__(self):
        pass
    def train(self,x_train,y_train,**kwargs):
        try:
            logging.info("Training Linear Regression Model")
            reg = LinearRegression(**kwargs)
            reg.fit(x_train,y_train)
            logging.info("Training Completed")  
            return reg
        except Exception as e:
            logging.error(f"Error in Training Linear Regression Model{e}")
            raise e