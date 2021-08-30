from pydantic import BaseModel

class CaptionGeneratorRequest(BaseModel):
    imageb64: str

class CaptionGeneratorResponse(BaseModel):
    caption: str
