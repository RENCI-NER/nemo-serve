import os

from fastapi import FastAPI
from src.ModelSingleton import ModelFactory
from src.ApiDataStructs import Query
import uvicorn
import logging
import yaml

app = FastAPI()
logger = logging.getLogger('gunicorn.error')


@app.on_event("startup")
def init_nlp_model():
    logger.info("Loading config file")
    config_file_path = os.environ.get('CONFIG_PATH',
                                      os.path.join(
                                          os.path.dirname(os.path.realpath(__file__)),
                                          "..",
                                          'config.yaml'
                                      )
                                      )
    logger.info(config_file_path)
    with open(config_file_path) as config_stream:
        config = yaml.load(config_stream, Loader=yaml.SafeLoader)
    logger.info(config)
    for model_name in config:
        cls = None
        gt_path = None
        logger.info(model_name)
        if model_name == 'token_classification':
            cls = ModelFactory.model_classes.get(config[model_name]['class'])
            path = config[model_name]['path']

        elif model_name == 'sapbert':
            path = config[model_name]['path']
            cls = ModelFactory.model_classes.get(config[model_name]['class'])
            gt_path = config[model_name]['ground_truth_data_path']
            gt_id_path = config[model_name]['ground_truth_data_ids_path']
            logger.info(f'path: {path}, cls: {cls}, gt_path: {gt_path}, gt_id_path: {gt_id_path}')
        logger.info(cls)
        if cls is None:
            raise ValueError(
                f"model class {config[model_name]['class']} not found please use one of {ModelFactory.model_classes.keys()}, "
                f"Or add your wrapper to ModelFactory.model_classes dictionary")
        if gt_path:
            logger.info('before loading sapbert model')
            ModelFactory.load_model(name=model_name, path=path, model_class=cls, ground_truth_data_path=gt_path,
                                    ground_truth_data_ids_path=gt_id_path)
            logger.info('after loading sapbert model')
        else:
            ModelFactory.load_model(name=model_name, path=path, model_class=cls)
        logger.info(f"Loaded {cls} model from {path} as {model_name}")


@app.post("/annotate/")
async def annotate_text(query: Query):
    if query.model_name not in ModelFactory.get_model_names():
        return {"error": f"please provide a valid model name from {ModelFactory.get_model_names()}"}, 400
    return ModelFactory.query_model(model_name=query.model_name, query_text=query.text)


@app.get("/models/")
async def get_model_names():
    """
    """
    return ModelFactory.get_model_names()


if __name__ == '__main__':
    uvicorn.run(app, port=8080)
