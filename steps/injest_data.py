from zenml import step
import logging
import pandas as pd

class InjestData:
    def __init__(self, data_path:str):
        self.data_path = data_path
    def get_data(self):
        logging.info(f"Injesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def injest_Data(data_path : str) -> pd.DataFrame :
        try:
            injestdata=InjestData(data_path)
            df=injestdata.get_data()
            return df
        except Exception as e:
            logging.error(f"Failed to injest data from {data_path} with error {e}")
