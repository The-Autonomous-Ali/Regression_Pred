import logging
import sys
import os
from typing import Optional
from pandas import DataFrame
from insurance_structure.constant.training_pipeline import SCHEMA_FILE_PATH
from insurance_structure.entity.config_entity import StrokePredictorConfig
from insurance_structure.entity.s3_estimator import InsurancePriceEstimator
from insurance_structure.exception import InsurancePriceException
from insurance_structure.logger import logging
from insurance_structure.utils.main_utils import read_yaml_file

class HeartData:
    def __init__(self, 
                 sex: str,
                 age: int,
                 bmi: float,
                 children: int,
                 smoker: str,
                 region: str,
                 charges: float):
        """
        Heart Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.sex = sex
            self.age = age
            self.bmi = bmi
            self.children = children
            self.smoker = smoker
            self.region = region
            self.charges = charges
        except Exception as e:
            raise InsurancePriceException(e, sys) from e

    def get_heart_stroke_input_data_frame(self) -> DataFrame:
        """
        This function returns a DataFrame from HeartData class input
        """
        try:
            heart_stroke_input_dict = self.get_heart_stroke_data_as_dict()
            return DataFrame(heart_stroke_input_dict)
        except Exception as e:
            raise InsurancePriceException(e, sys) from e

    def get_heart_stroke_data_as_dict(self) -> dict:
        """
        This function returns a dictionary from HeartData class input 
        """
        try:
            input_data = {
                "sex": [self.sex],
                "age": [self.age],
                "bmi": [self.bmi],
                "children": [self.children],
                "smoker": [self.smoker],
                "region": [self.region],
                "charges": [self.charges]
            }
            return input_data
        except Exception as e:
            raise InsurancePriceException(e, sys)

class HeartStrokeClassifier:
    def __init__(self, 
                 prediction_pipeline_config: StrokePredictorConfig = StrokePredictorConfig()) -> None:
        """
        :param prediction_pipeline_config:
        """
        try:
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
            self.use_aws = os.getenv('USE_AWS', 'False').lower() == 'true'
        except Exception as e:
            raise InsurancePriceException(e, sys)

    def predict(self, dataframe: DataFrame) -> str:
        """
        This method of HeartStrokeClassifier returns a regression prediction
        Returns: Prediction as a string
        """
        try:
            logging.info("Entered predict method of HeartStrokeClassifier class")

            if self.use_aws:
                # AWS interaction code
                model = InsurancePriceEstimator(
                    bucket_name=self.prediction_pipeline_config.model_bucket_name,
                    model_path=self.prediction_pipeline_config.model_file_path,
                )
                result = model.predict(dataframe)
                # Return the predicted value
                return f"Predicted value: {result}"
            else:
                # Mock prediction for local testing
                # Simulate a regression output, e.g., a random value or a fixed value
                mock_prediction = 15000.0  # Replace this with actual local model prediction
                return f"Mock predicted value: {mock_prediction}"

        except Exception as e:
            raise InsurancePriceException(e, sys)
