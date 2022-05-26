import os

from fastapi import FastAPI
from .ModelSingleton import ModelFactory, TokenClassificationModelWrapper
from .ApiDataStructs import Query
import uvicorn
import logging

app = FastAPI()
logger = logging.Logger("server")


@app.on_event("startup")
def init_nlp_model():
    logger.info("Loading nlp model")
    path = os.environ.get("MODEL_PATH")
    ModelFactory.load_model(name="token_classification", path=path, model_class=TokenClassificationModelWrapper)
    logger.info(f"Loaded token_classification model from {path}")


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
