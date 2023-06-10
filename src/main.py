from fastapi import FastAPI
from api import (
    doraemong,
)
import uvicorn
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.include_router(doraemong)

app.mount("/", StaticFiles(directory="static", html=True), name="index")
app.mount("/doraemong", StaticFiles(directory="static/doraemong",
          html=True), name="doraemong")

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.0.14", port=2030)
