import os

from fastapi import FastAPI
from src.ModelSingleton import ModelFactory, init_models
from src.ApiDataStructs import Query
import uvicorn
import logging

app = FastAPI()
logger = logging.getLogger('gunicorn.error')


@app.on_event("startup")
def init_nlp_model():
    config_file_path = os.environ.get('CONFIG_PATH',
                                      os.path.join(
                                          os.path.dirname(os.path.realpath(__file__)),
                                          "..",
                                          'config.yaml'
                                      )
                                      )
    init_models(config_file_path)

@app.post("/annotate/")
async def annotate_text(query: Query):
    if query.model_name not in ModelFactory.get_model_names():
        return {"error": f"please provide a valid model name from {ModelFactory.get_model_names()}"}, 400
    return ModelFactory.query_model(model_name=query.model_name, query_text=query.text, query_count=query.count, **query.args)


@app.get("/models/")
async def get_model_names():
    """
    """
    return ModelFactory.get_model_names()


if __name__ == '__main__':
    uvicorn.run(app, port=8080)
