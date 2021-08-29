from fastapi import FastAPI
import uvicorn
# from routers.captiongenerator import fileCount
# from routers.cartoonizer import filesUsingLoop
from routers.textsummarizer import summarization

app = FastAPI()

app.include_router(summarization.router)

if __name__ == '__main__':
    uvicorn.run("main:app", port=8000, host='127.0.0.1', reload=True)
