from fastapi import FastAPI
from api import (
    doraemong,
    idol_position_router,
)
import uvicorn
from fastapi.staticfiles import StaticFiles
app = FastAPI()


for router in [doraemong, idol_position_router]:
    app.include_router(router)
    static_directory = f"static/{router.prefix}"
    app.mount(f"/{router.prefix}", StaticFiles(directory=static_directory,
              html=True), name=router.prefix)

app.mount("/", StaticFiles(directory="static", html=True), name="index")

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.0.14", port=2030)
