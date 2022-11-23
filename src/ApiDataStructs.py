from pydantic import BaseModel

class Query(BaseModel):
    text: str
    model_name: str
    # count is not required and will default to 1 if nothing is passed
    count: int = 1

    class Config:
        schema_extra = {
            "example": {
                "text": "input text for prediction",
                "model_name": "tokenclassification or sapbert",
                "count": 1
            }
        }
