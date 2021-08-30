# from config.config import Configuration
from typing import Optional, List
from fastapi import FastAPI, Query, APIRouter, status
from pydantic import BaseModel, DirectoryPath
import os
import uvicorn
import base64
from io import BytesIO
from PIL import Image
from utils.helpers import clearFolderContents
from mlmodels import Cartoonizer
from datamodels import MessageModel, CartoonizerRequest, CartoonizerResponse
from config.config import HOME_DIR
router = APIRouter(tags = ["Cartoonization using Cartoon-GAN by Filip Anderson"])

clearFolderContents(HOME_DIR)
pretrained_dir = "./checkpoints/trained_netG.pth"
cartoonizermodel = Cartoonizer(pretrained_dir)

@router.post("/cartoonization", response_model=CartoonizerResponse, summary="Cartoonize image using Cartoon-GAN",status_code=status.HTTP_200_OK, responses={404: {"model": MessageModel}})
def cartoonize(cartoonizerrequest: CartoonizerRequest):
    imageb64 = cartoonizerrequest.imageb64
    # https://jdhao.github.io/2020/03/17/base64_opencv_pil_image_conversion/
    # convert the base64 image string to PIL image
    # 1. convert the base64 image string to file-like object
    # 2. convert the file-like object to a PIL Image Object
    
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

    # output_path is the path where we will temporarily store the cartoonized image 
    output_path = os.path.join(HOME_DIR, "output.png")
    
    # our model returns a normal PIL.Image object, which is stored in the variable cartoonized_image
    cartoonized_image = cartoonizermodel.cartoonize(input_path, output_path)

    # now, we create (instantiate) a file-like object, using BytesIO()
    # we could either store the cartoonized_image variable (PIL.Image object) to a file, using the .save() method, if a path was provided
    # or we can save that cartoonized_image variable (PIL.Image object) to a file-like object
    cartoonized_file_object = BytesIO()

    # the PIL.Image object has the method save, which can either save the file to the disk, if a path is provided
    # or to a file-like object if that is provided, here we do the latter
    cartoonized_image.save(cartoonized_file_object, format="JPEG")

    # now, we will convert the file-like object (cartoonized_file_object) to the byte representation (cartoonized_bytes)
    cartoonized_bytes = cartoonized_file_object.getvalue()

    # after getting the byte representation (cartoonized_bytes), we will convert it to base64 representation (cartoonized_imageb64)
    # so that we can return it over the internet
    cartoonized_imageb64 = base64.b64encode(cartoonized_bytes)
    
    # we will create a dictionary, with keys as the properties of the response data model, and values as the values of the response data model
    cartoonized_dict = {"cartoonized_imageb64": cartoonized_imageb64}

    # create the response model object from the dictionary
    cartoonizerresponse = CartoonizerResponse(**cartoonized_dict)

    return cartoonizerresponse


