from fastapi import FastAPI
from api import doraemong, IdolPosition

import uvicorn
from fastapi.staticfiles import StaticFiles
app = FastAPI()

routers = ["doraemong", "IdolPosition"]

for router in routers:
    print(router)
    exec(f"app.include_router({router})")
    app.mount(f"/{router}", StaticFiles(directory=f"static/{router}",
              html=True), name=router)

app.mount("/", StaticFiles(directory="static", html=True), name="index")

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.0.14", port=2030)
