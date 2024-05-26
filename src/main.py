from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from api import * 

app = FastAPI()

routers = ["balloonfist_router", "idolposition_router","faceage_router", "whostheking_router"]

allowed_origins = [
    "http://localhost:2030",
    "http://127.0.0.1:2030",
    "https://tensecgames.com",
    "http://syg0203.iptime.org:2030",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def access_control_middleware(request: Request, call_next):
    if request.url.path in ['/redoc']:
        return JSONResponse(status_code=403, content={"detail": "Access denied"})
    
    if any(request.url.path.startswith(f"/{router}") for router in routers):
        referer = request.headers.get('referer')
        if not referer or not any(origin in referer for origin in allowed_origins):
            return JSONResponse(status_code=403, content={"detail": "Access denied"})
        
    response = await call_next(request)
    return response

# app.add_middleware(
#     TrustedHostMiddleware, allowed_hosts=["tensecgames.com","*.tensecgames.com"] 
# )

for router in routers:
    exec(f"app.include_router({router})")

app.mount("/", StaticFiles(directory="static", html=True), name="index")

if __name__ == "__main__":
    print("os.name :::::: ",os.name)
    if os.name == 'posix':
        print("os : ", os.name, "linux:gunicorn")
        os.system("gunicorn -c ./gunicorn.config.py")
    elif os.name == 'nt':
        import uvicorn
        print("os.name :::::: ", os.name, "windows:uvicorn")
        uvicorn.run(app, host="0.0.0.0", port=2030)