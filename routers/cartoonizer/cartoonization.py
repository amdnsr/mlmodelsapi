from config.config import Configuration
from typing import Optional, List
from fastapi import FastAPI, Query, APIRouter, status
from pydantic import BaseModel, DirectoryPath
import os
import uvicorn

from mlmodels import Cartoonizer
from datamodels import MessageModel, CartoonizerRequest, CartoonizerResponse

router = APIRouter(tags = ["Text Summarization using Facebook BART"])

pretrained_dir = "../cartoonganapi/project/checkpoints/trained_netG.pth"
cartoonizermodel = Cartoonizer.Cartoonizer(pretrained_dir)
# textsummarizermodel = TextSummarizer.TextSummarizer(model_directory)

@router.post("/cartoonization", response_model=CartoonizerResponse, summary="Cartoonize image using Cartoon-GAN",status_code=status.HTTP_200_OK, responses={404: {"model": MessageModel}})
def cartoonize(cartoonizerrequest: CartoonizerRequest):
    image = cartoonizerrequest.image
    cartoonized_image = cartoonizermodel.cartoonize(image)
    cartoonized_dict = {"cartoonized_image": cartoonized_image}
    cartoonizerresponse = CartoonizerResponse(**cartoonized_image)
    return cartoonizerresponse


