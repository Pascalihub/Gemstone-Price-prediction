from src.gemstone.config.configuration import ConfigurationManager
from src.gemstone.components.data_ingestion import DataIngestion
from src.gemstone.components.data_transformation import DataTransformation
from src.gemstone.components.data_validation import DataValiadtion
from src.gemstone.logger import logging


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValiadtion(config=data_validation_config)
        data_validation.validate_all_columns()