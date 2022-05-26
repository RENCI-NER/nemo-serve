from pydantic import BaseModel


class Query(BaseModel):
    text: str
    model_name: str