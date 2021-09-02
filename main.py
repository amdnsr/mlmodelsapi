import config
from utils.helpers import createFolder
from fastapi import FastAPI
import uvicorn
from routers.captiongenerator.captiongeneration import router as captiongeneration_router
from routers.cartoonizer.cartoonization import router as cartoonization_router
from routers.textsummarizer.summarization import router as summarization_router
from config.config import HOME_DIR

app = FastAPI()

app.include_router(captiongeneration_router)
app.include_router(cartoonization_router)
app.include_router(summarization_router)

if __name__ == '__main__':
    createFolder(HOME_DIR)
    uvicorn.run("main:app", port=8000, host='127.0.0.1', reload=True)
