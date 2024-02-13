from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from api import * 

app = FastAPI()

routers = ["balloonfist_router", "idolposition_router","faceage_router", "whostheking_router"]


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
    print("os.name :::::: ",os.name)
    if os.name == 'posix':
        print("os : ", os.name, "linux:gunicorn")
#        os.system("gunicorn -w 1 -k uvicorn.workers.UvicornWorker src.main:app -b 0.0.0.0:2030 --timeout 30 --keep-alive 30 --max-requests 200")
#        os.system("gunicorn -w 1 -k uvicorn.workers.UvicornWorker src.main:app -b 0.0.0.0:2030 --timeout 30 --keep-alive 30 --max-requests 200  --log-config ./logs/config/uvicorn_log.ini")
        os.system("gunicorn -c ./gunicorn.config.py")
    elif os.name == 'nt':
        import uvicorn
        print("os.name :::::: ", os.name, "windows:uvicorn")
        uvicorn.run(app, host="0.0.0.0", port=2030)



