
from typing import List
from datetime import datetime
import numpy as np
import pandas as pd
import warnings

import joblib

from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

def get_period_day(date):
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()
    evening_max = datetime.strptime("23:59", '%H:%M').time()
    night_min = datetime.strptime("00:00", '%H:%M').time()
    night_max = datetime.strptime("4:59", '%H:%M').time()
    
    if(date_time > morning_min and date_time < morning_max):
        return 'mañana'
    elif(date_time > afternoon_min and date_time < afternoon_max):
        return 'tarde'
    elif(
        (date_time > evening_min and date_time < evening_max) or
        (date_time > night_min and date_time < night_max)
    ):
        return 'noche'
    
def is_high_season(fecha):
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
    range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
    range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
    range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
    
    if ((fecha >= range1_min and fecha <= range1_max) or 
        (fecha >= range2_min and fecha <= range2_max) or 
        (fecha >= range3_min and fecha <= range3_max) or
        (fecha >= range4_min and fecha <= range4_max)):
        return 1
    else:
        return 0
    
def get_min_diff(data):
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds())/60
    return min_diff

class DelayModel:

    def __init__(
        self
    ):
        self._model = None
        self.top_10_features = [
                "OPERA_Latin American Wings", 
                "MES_7",
                "MES_10",
                "OPERA_Grupo LATAM",
                "MES_12",
                "TIPOVUELO_I",
                "MES_4",
                "MES_11",
                "OPERA_Sky Airline",
                "OPERA_Copa Air"
            ]

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ):
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        self.threshold_in_minutes = 15
        
        self._model = LogisticRegression(class_weight = "balanced") 
        
        self.data = data
        self.data['period_day'] = data['Fecha-I'].apply(get_period_day)
        self.data['high_season'] = data['Fecha-I'].apply(is_high_season)
        self.data['min_diff'] = data.apply(get_min_diff, axis = 1)
        
        self.data['delay'] = np.where(data['min_diff'] > self.threshold_in_minutes, 1, 0)
        
        features = pd.concat([
                pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
                pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
                pd.get_dummies(data['MES'], prefix = 'MES')], 
                axis = 1
            )
        
        selected_features = features[self.top_10_features]

        if target_column is not None:
            # If target_column is specified, return both features and target
            target = self.data[target_column].to_frame()
            return selected_features, target
        else:
            # If target_column is not specified, return just the features
            return selected_features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        self._model.fit(features, target)
        
        return self._model

    def predict(
        self,
        features: pd.DataFrame
        ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        # Make predictions using the trained model and adjusted threshold
        predictions = self._model.predict(features)
        return predictions.tolist()
    
    def save_model(self, filename: str):
        """
        Save the trained model to a file using Joblib.

        Args:
            filename (str): Name of the file to save the model.
        """
        joblib.dump(self._model, filename)

    def load_model(self, filename: str):
        """
        Load a trained model from a file using Joblib.

        Args:
            filename (str): Name of the file containing the saved model.
        """
        self._model = joblib.load(filename)
