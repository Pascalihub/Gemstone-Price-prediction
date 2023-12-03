from src.gemstone.pipeline.st_01_data_ingestion import DataIngestionTrainingPipeline
from src.gemstone.pipeline.st_02_data_validation import DataValidationTrainingPipeline
from src.gemstone.pipeline.st_03_data_transformation import DataTransformationTrainingPipeline
from src.gemstone.pipeline.st_04_model_trainer import ModelTrainerTrainingPipeline
from src.gemstone.pipeline.st_o5_model_evaluation import ModelEvaluationTrainingPipeline
from src.gemstone.logger import logging

# Data Ingestion Stage
STAGE_NAME = "Data Ingestion stage"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e

# Data Validation Stage
STAGE_NAME = "Data Validation stage"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e

# Data Transformation Stage
STAGE_NAME = "Data Transformation stage"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e

# Model Trainer Stage
STAGE_NAME = "Model Trainer stage"
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e



STAGE_NAME = "Data evaluation stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = ModelEvaluationTrainingPipeline()
   data_ingestion.main()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise e