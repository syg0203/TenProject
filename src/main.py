from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from api import * 

app = FastAPI()

semaphore = asyncio.Semaphore(4)

class ConcurrencyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_concurrent_requests: int):
        super().__init__(app)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def dispatch(self, request: Request, call_next):
        async with self.semaphore:
            response = await call_next(request)
        return response

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

app.add_middleware(ConcurrencyMiddleware, max_concurrent_requests=4)

if __name__ == "__main__":
    print("os.name :::::: ",os.name)
    if os.name == 'posix':
        print("os : ", os.name, "linux:gunicorn")
        os.system("gunicorn -c ./modules/core/gunicorn.config.py")
    elif os.name == 'nt':
        import uvicorn
        print("os.name :::::: ", os.name, "windows:uvicorn")
        uvicorn.run(app, host="0.0.0.0", port=2030)
