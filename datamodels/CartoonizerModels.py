from pydantic import BaseModel

class CartoonizerRequest(BaseModel):
    image: str

class CartoonizerResponse(BaseModel):
    cartoonized_image: str
