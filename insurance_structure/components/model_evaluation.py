from insurance_structure.entity.config_entity import ModelEvaluationConfig
from insurance_structure.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from insurance_structure.utils.main_utils import load_object
from sklearn.metrics import r2_score
from insurance_structure.exception import InsurancePriceException
from insurance_structure.constant.training_pipeline import TARGET_COLUMN
from insurance_structure.logger import logging
import os, sys
import pandas as pd
from typing import Dict, Optional
from insurance_structure.entity.s3_estimator import InsurancePriceEstimator
from dataclasses import dataclass
from insurance_structure.entity.estimator import InsurancePredModel

@dataclass
class EvaluateModelResponse:
    trained_model_r2_score: float
    best_model_r2_score: Optional[float]
    is_model_accepted: bool
    difference: float

class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        """
        :param model_eval_config: Output reference of data evaluation artifact stage
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param model_trainer_artifact: Output reference of model_trainer_artifact stage
        """
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise InsurancePriceException(e, sys) from e

    def get_best_model(self) -> Optional[InsurancePriceEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get the model in production
        
        Output      :   Returns model object if available in S3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            # Check if AWS credentials are set
            aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            
            if not aws_access_key_id or not aws_secret_access_key:
                logging.warning("AWS credentials are not set. Unable to retrieve the best model.")
                return None

            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            insurance_price_estimator = InsurancePriceEstimator(bucket_name=bucket_name, model_path=model_path)

            if insurance_price_estimator.is_model_present(model_path=model_path):
                return insurance_price_estimator
            return None
        except Exception as e:
            logging.error(f"Error occurred in get_best_model: {e}")
            return None

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function evaluates the trained model against the best model in production
        """
        try:
            # Read test data
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x_test, y_test = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            # Load the trained model
            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)

            # Get predictions from the trained model
            y_pred_trained = trained_model.predict(x_test)

            # Calculate regression metrics for the trained model
            trained_model_r2_score = r2_score(y_test, y_pred_trained)

            # Evaluate the best model
            best_model_r2_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                y_pred_best_model = best_model.predict(x_test)
                best_model_r2_score = r2_score(y_test, y_pred_best_model)

            # Compare models and decide whether to accept the new model
            tmp_best_model_score = -float('inf') if best_model_r2_score is None else best_model_r2_score
            is_model_accepted = trained_model_r2_score > tmp_best_model_score
            difference = trained_model_r2_score - tmp_best_model_score

            result = EvaluateModelResponse(
                trained_model_r2_score=trained_model_r2_score,
                best_model_r2_score=best_model_r2_score,
                is_model_accepted=is_model_accepted,
                difference=difference
            )

            logging.info(f"Evaluation Result: {result}")
            return result

        except Exception as e:
            logging.error(f"Error occurred in evaluate_model: {e}")
            raise InsurancePriceException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function initiates the model evaluation process
        
        Output      :   Returns a ModelEvaluationArtifact object with the evaluation result
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            evaluate_model_response = self.evaluate_model()

            s3_model_path = self.model_eval_config.s3_model_key_path
                
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            logging.error(f"Error occurred in initiate_model_evaluation: {e}")
            raise InsurancePriceException(e, sys) from e
