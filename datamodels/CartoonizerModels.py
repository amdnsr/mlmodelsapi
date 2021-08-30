from pydantic import BaseModel

class CartoonizerRequest(BaseModel):
    imageb64: str

class CartoonizerResponse(BaseModel):
    cartoonized_imageb64: str
