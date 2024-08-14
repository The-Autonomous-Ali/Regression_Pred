import pymongo
import logging
import os
from insurance_structure.constant.env_variable import MONGODB_URL_KEY
from insurance_structure.configuration.mongo_db_connection import ca


def test_connection():
    try:
        mongo_db_url = os.getenv(MONGODB_URL_KEY)
        logging.info(f"Connecting to MongoDB at {mongo_db_url}")
        client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
        # Attempt to list databases to confirm connection
        client.list_database_names()
        logging.info("Connection successful")
    except Exception as e:
        logging.error(f"Failed to connect: {e}")

test_connection()
