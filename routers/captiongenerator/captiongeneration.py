from utils.helpers import clearFolderContents
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
import argparse
import base64
from io import BytesIO

from config.config import HOME_DIR
from mlmodels import CaptionGenerator
from datamodels import MessageModel, CaptionGeneratorRequest, CaptionGeneratorResponse

router = APIRouter(tags = ["Caption Generation using Xception and another model."])

max_length = 32
tokenizer = load(open("assets/captiongenerator/tokenizer.p","rb"))
model = load_model('checkpoints/captiongenerator/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")
clearFolderContents(HOME_DIR)

captiongeneratormodel = CaptionGenerator(caption_model=model, feature_model=xception_model, tokenizer=tokenizer, max_length=max_length)

@router.post("/captiongeneration", response_model=CaptionGeneratorResponse, summary="Caption for the input image.",status_code=status.HTTP_200_OK, responses={404: {"model": MessageModel}})
def generatecaption(captiongeneratorrequest: CaptionGeneratorRequest):
    imageb64 = captiongeneratorrequest.imageb64

    # This will give the byte representation of the imageb64
    image_bytes = base64.b64decode(imageb64)
    
    # This will convert the byte representation to a file-like object, which can be opened by PIL.Image
    # (PIL.Image.open takes either the file path or a file-like object)
    image_file_object = BytesIO(image_bytes)  # convert image to file-like object
    
    # Now, PIL.Image.open loads the file-like object as a PIL Image object
    # i.e. a normal image, which can be saved or show using image.show()
    image = Image.open(image_file_object)   # image is now PIL Image object

    # input_path is the path where we will temporarily store the input image 
    input_path = os.path.join(HOME_DIR, "input.png")
    image.save(input_path)

    # get the caption for the input image using the captiongeneratormodel
    caption = captiongeneratormodel.get_description(input_path)

    # we will create a dictionary, with keys as the properties of the response data model, and values as the values of the response data model
    caption_dict = {"caption": caption}

    # create the response model object from the dictionary
    captiongeneratorresponse = CaptionGeneratorResponse(**caption_dict)

    return captiongeneratorresponse


