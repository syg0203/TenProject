from fastapi import FastAPI
<<<<<<< HEAD
from api import doraemong, IdolPosition
=======
from api import (
    doraemong,
    IdolPosition,
)
>>>>>>> origin
import uvicorn
from fastapi.staticfiles import StaticFiles
app = FastAPI()

<<<<<<< HEAD
routers = ["doraemong", "IdolPosition"]
=======
app.include_router(doraemong)
app.include_router(IdolPosition)
>>>>>>> origin

for router in routers:
    print(router)
    exec(f"app.include_router({router})")
    app.mount(f"/{router}", StaticFiles(directory=f"static/{router}",
              html=True), name=router)

app.mount("/", StaticFiles(directory="static", html=True), name="index")
<<<<<<< HEAD
=======
app.mount("/doraemong", StaticFiles(directory="static/doraemong",
          html=True), name="doraemong")
app.mount("/IdolPosition", StaticFiles(directory="static/IdolPosition",
          html=True), name="IdolPosition")

>>>>>>> origin

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.0.14", port=2030)
