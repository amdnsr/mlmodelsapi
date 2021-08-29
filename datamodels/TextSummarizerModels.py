from pydantic import BaseModel

class TextSummarizerRequest(BaseModel):
    text: str

class TextSummarizerResponse(BaseModel):
    summary: str
