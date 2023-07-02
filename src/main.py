from fastapi import FastAPI
from api import (
    doraemong,
    idol_position_router,
)
import uvicorn
from fastapi.staticfiles import StaticFiles
app = FastAPI()

app.include_router(doraemong)
app.include_router(idol_position_router)


app.mount("/", StaticFiles(directory="static", html=True), name="index")
app.mount("/doraemong", StaticFiles(directory="static/doraemong",
          html=True), name="doraemong")
app.mount("/Idol_position_test", StaticFiles(directory="static/Idol_position_test",
          html=True), name="Idol_position_test")


if __name__ == "__main__":
    uvicorn.run(app, host="192.168.0.14", port=2030)
