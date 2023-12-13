import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, RMSE, R2
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
@step
def evaluate_model(model:RegressorMixin,
    X_test:pd.DataFrame ,
    y_test:pd.DataFrame,
                   ) -> Tuple[
                       Annotated[float, "r2"],
                       Annotated[float, "rmse"]
                   ]:
    '''
    Evaluate model performance
    Args:
        df: pandas DataFrame
    '''
    try:
        logging.info("Evaluating Model")
        prediction=model.predict(X_test)
        mse = MSE().calculate_scores(y_test,prediction)
        rmse  = RMSE().calculate_scores(y_test,prediction)
        r2 = R2().calculate_scores(y_test,prediction)

        return r2,rmse
    except Exception as e:
        logging.error(f"Error in Evaluating Model {e}")
        raise e
    
