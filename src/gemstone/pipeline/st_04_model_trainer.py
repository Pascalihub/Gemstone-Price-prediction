from src.gemstone.config.configuration import ConfigurationManager
from src.gemstone.components.data_ingestion import DataIngestion
from src.gemstone.components.data_transformation import DataTransformation
from src.gemstone.components.model_trainer import ModelTrainer
from src.gemstone.logger import logging


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.initiate_model_training()