from pydantic import BaseModel

class Query(BaseModel):
    text: str
    model_name: str
    # count is not required and will default to 1 if nothing is passed
    count: int = 1
