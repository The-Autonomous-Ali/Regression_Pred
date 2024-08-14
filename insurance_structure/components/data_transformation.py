import sys
from typing import Union

import numpy as np
import pandas as pd
from insurance_structure.constant.training_pipeline import TARGET_COLUMN, SCHEMA_FILE_PATH
from insurance_structure.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact
from insurance_structure.entity.config_entity import DataTransformationConfig
from insurance_structure.utils.main_utils import read_yaml_file, save_numpy_array_data, save_object
from insurance_structure.exception import InsurancePriceException
from insurance_structure.logger import logging
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: Configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise InsurancePriceException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise InsurancePriceException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object 
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            logging.info("Got numerical, categorical, transformation columns from schema config")
            
            numerical_columns = self._schema_config['Numerical_columns']
            categorical_columns = self._schema_config['Categorical_columns']
            transform_columns = self._schema_config['Transformation_columns']

            # Ensure that the target column (charges) is not included in the feature lists
            if TARGET_COLUMN in numerical_columns:
                numerical_columns.remove(TARGET_COLUMN)
            if TARGET_COLUMN in categorical_columns:
                categorical_columns.remove(TARGET_COLUMN)
            if TARGET_COLUMN in transform_columns:
                transform_columns.remove(TARGET_COLUMN)

            numeric_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            transform_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer(standardize=True))
            ])

            preprocessor = ColumnTransformer(
                [
                    ("Numeric_Pipeline", numeric_pipeline, numerical_columns),
                    ("Categorical_Pipeline", categorical_pipeline, categorical_columns),
                    ("Power_Transformation", transform_pipe, transform_columns)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")

            return preprocessor

        except Exception as e:
            raise InsurancePriceException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")

                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                logging.info(f"Columns in training data: {train_df.columns.tolist()}")
                logging.info(f"Columns in testing data: {test_df.columns.tolist()}")

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info(f"Columns in training data before transformation: {input_feature_train_df.columns.tolist()}")

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                logging.info(f"Columns in testing data before transformation: {input_feature_test_df.columns.tolist()}")

                logging.info("Applying preprocessing object on training dataframe and testing dataframe")

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                logging.info(f"Transformed data shape for training: {input_feature_train_arr.shape}")
                
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)
                logging.info(f"Transformed data shape for testing: {input_feature_test_arr.shape}")

                # Remove SMOTEENN application for regression
                input_feature_train_final = input_feature_train_arr
                target_feature_train_final = target_feature_train_df
                input_feature_test_final = input_feature_test_arr
                target_feature_test_final = target_feature_test_df

                # Save transformed data and preprocessor object
                train_arr = np.c_[
                    input_feature_train_final, np.array(target_feature_train_final)
                ]
                test_arr = np.c_[
                    input_feature_test_final, np.array(target_feature_test_final)
                ]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object")

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)
        except Exception as e:
            raise InsurancePriceException(e, sys) from e
