import sys
from typing import Tuple

import numpy as np
from insurance_structure.entity.artifact_entity import (DataTransformationArtifact, ModelTrainerArtifact)
from insurance_structure.entity.config_entity import ModelTrainerConfig
from insurance_structure.exception import InsurancePriceException
from insurance_structure.logger import logging
from insurance_structure.utils.main_utils import load_numpy_array_data, load_object, save_object
from neuro_mf import ModelFactory
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from insurance_structure.entity.estimator import InsurancePredModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, dict]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function uses neuro_mf to get the best model object and report of the best model
        
        Output      :   Returns model trainer artifact object and best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Using neuro_mf to get best model object and report")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

            best_model_detail = model_factory.get_best_model(
                X=x_train, y=y_train, base_accuracy=self.model_trainer_config.expected_score
            )
            model_obj = best_model_detail.best_model

            y_pred = model_obj.predict(x_test)
            
            # Regression metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            metric_artifact = {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2
            }
            
            logging.info(f"Regression Metrics - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")

            return best_model_detail, metric_artifact
        
        except Exception as e:
            raise InsurancePriceException(e, sys) from e
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            
            best_model_detail, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            if best_model_detail.best_score < self.model_trainer_config.expected_score:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")

            heart_stroke_model = InsurancePredModel(preprocessing_object=preprocessing_obj,
                                                   trained_model_object=best_model_detail.best_model)
            logging.info("Created Heart Stroke object with preprocessor and model")
            logging.info("Saving the best model to file")
            save_object(self.model_trainer_config.trained_model_file_path, heart_stroke_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise InsurancePriceException(e, sys) from e
