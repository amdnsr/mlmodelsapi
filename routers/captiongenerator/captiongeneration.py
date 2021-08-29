from config.config import Configuration
from typing import Optional, List
from fastapi import FastAPI, Query, APIRouter, status
from pydantic import BaseModel, DirectoryPath
import os
import uvicorn


import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse


from mlmodels import CaptionGenerator
from datamodels import MessageModel, CaptionGeneratorRequest, CaptionGeneratorResponse

router = APIRouter(tags = ["Caption Generation using Xception and another model."])

max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('checkpoints/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")
# captiongeneratormodel = None
captiongeneratormodel = CaptionGenerator.CaptionGenerator(caption_model=model, feature_model=xception_model, tokenizer=tokenizer, max_length=max_length)

@router.post("/captiongeneration", response_model=CaptionGeneratorResponse, summary="Caption for the input image.",status_code=status.HTTP_200_OK, responses={404: {"model": MessageModel}})
def generatecaption(captiongeneratorrequest: CaptionGeneratorRequest):
    image = captiongeneratorrequest.image
    caption = captiongeneratormodel.get_description(image)
    caption_dict = {"caption": caption}
    captiongeneratorresponse = CaptionGeneratorResponse(**caption_dict)
    return captiongeneratorresponse


