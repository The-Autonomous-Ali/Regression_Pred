import os
import sys
import certifi
import pymongo
from pymongo import MongoClient
from insurance_structure.constant.database import DATABASE_NAME
from insurance_structure.constant.env_variable import MONGODB_URL_KEY
from insurance_structure.exception import InsurancePriceException
from insurance_structure.logger import logging

ca = certifi.where()

class MongoDBClient:
    """
    Class Name :   MongoDBClient
    Description :   This class connects to MongoDB and provides access to the database.
    """
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment key: {MONGODB_URL_KEY} is not set.")
                logging.info(f"Connecting to MongoDB at {mongo_db_url}")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except pymongo.errors.ServerSelectionTimeoutError as e:
            logging.error(f"Server Selection Timeout Error: {e}")
            raise InsurancePriceException(e, sys) from e
        except pymongo.errors.ConnectionError as e:
            logging.error(f"Connection Error: {e}")
            raise InsurancePriceException(e, sys) from e
        except Exception as e:
            logging.error(f"General Error: {e}")
            raise InsurancePriceException(e, sys) from e
