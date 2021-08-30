import config
from utils.helpers import createFolder
from fastapi import FastAPI
import uvicorn
from routers.captiongenerator import captiongeneration
from routers.cartoonizer import cartoonization
from routers.textsummarizer import summarization
from config.config import HOME_DIR

app = FastAPI()

app.include_router(captiongeneration.router)
app.include_router(cartoonization.router)
app.include_router(summarization.router)

if __name__ == '__main__':
    createFolder(HOME_DIR)
    uvicorn.run("main:app", port=8000, host='127.0.0.1', reload=True)
