from config.config import Configuration
from typing import Optional, List
from fastapi import FastAPI, Query, APIRouter, status
from pydantic import BaseModel, DirectoryPath
import os
import uvicorn

from mlmodels import TextSummarizer
from datamodels import MessageModel, TextSummarizerRequest, TextSummarizerResponse

router = APIRouter(tags = ["Text Summarization using Facebook BART"])

model_directory = "../textsummarization/project/my_model_directory"
textsummarizermodel = None
# textsummarizermodel = TextSummarizer.TextSummarizer(model_directory)

@router.post("/textsummarization", response_model=TextSummarizerResponse, summary="Summary of the text using Facebook BART",status_code=status.HTTP_200_OK, responses={404: {"model": MessageModel}})
def summarize(textsummarizerrequest: TextSummarizerRequest):
    text = textsummarizerrequest.text
    summary = textsummarizermodel.get_summary(text)
    summary_dict = {"summary": summary}
    textsummarizerresponse = TextSummarizerResponse(**summary_dict)
    return textsummarizerresponse


