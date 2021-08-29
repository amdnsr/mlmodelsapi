from pydantic import BaseModel

class CaptionGeneratorRequest(BaseModel):
    image: str

class CaptionGeneratorResponse(BaseModel):
    caption: str
