from src.gemstone.config.configuration import ConfigurationManager
from src.gemstone.components.data_ingestion import DataIngestion
from src.gemstone.components.data_transformation import DataTransformation
from src.gemstone.components.model_trainer import ModelTrainer
from src.gemstone.logger import logging

STAGE_NAME = "Data transformation stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.initiate_model_training()

if __name__ == '__main__':   
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        data_ingestion = ModelTrainerTrainingPipeline()
        data_ingestion.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e