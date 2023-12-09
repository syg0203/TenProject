from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import uvicorn


from api import doraemong, IdolPosition, faceage, whostheking

app = FastAPI()

routers = ["doraemong", "IdolPosition","faceage", "whostheking"]


@app.middleware("http")
async def access_control_middleware(request: Request, call_next):
    if request.url.path in ['/redoc', '/docs']:
        return JSONResponse(status_code=403, content={"detail": "Access denied"})

    response = await call_next(request)
    return response

for router in routers:
    exec(f"app.include_router({router})")

app.mount("/", StaticFiles(directory="static", html=True), name="index")

if __name__ == "__main__":
    print("os.name",os.name)
    if os.name == 'posix':
        print("os : ", os.name, "linux:gunicorn")
        os.system(f"gunicorn -w {os.cpu_count()} -k uvicorn.workers.UvicornWorker main:app --host 0.0.0.0 --port 2030 --keep-alive 20 --max-requests 100")
    elif os.name == 'nt':
        print("os : ", os.name, "windows:uvicorn")
        uvicorn.run(app, host="0.0.0.0", port=2030)
