from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from api import doraemong, IdolPosition

app = FastAPI()

routers = ["doraemong", "IdolPosition"]

@app.middleware("http")
async def access_control_middleware(request: Request, call_next):
    print(f"Received request: {request.method} {request.url}")
    if request.url.path in ['/redoc','/docs']:
        return JSONResponse(status_code=403, content={"detail": "Access denied"})

    response = await call_next(request)
    return response

for router in routers:
    exec(f"app.include_router({router})")
    app.mount(f"/{router}", StaticFiles(directory=f"static/{router}",
              html=True), name=router)

app.mount("/", StaticFiles(directory="static", html=True), name="index")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2030)