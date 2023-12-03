from src.gemstone.config.configuration import ConfigurationManager
from src.gemstone.components.data_ingestion import DataIngestion
from src.gemstone.components.data_transformation import DataTransformation
from src.gemstone.components.model_trainer import ModelTrainer
from src.gemstone.components.model_evaluation import ModelEvaluation
from src.gemstone.logger import logging

STAGE_NAME = "Data evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.log_into_mlflow()

if __name__ == '__main__':   
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        data_ingestion = ModelEvaluationTrainingPipeline()
        data_ingestion.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e