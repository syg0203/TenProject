from fastapi import FastAPI, Request
from api import doraemong, IdolPosition
from fastapi.responses import JSONResponse

import uvicorn
from fastapi.staticfiles import StaticFiles
app = FastAPI()

routers = ["doraemong", "IdolPosition"]


@app.middleware("http")
async def access_control_middleware(request: Request, call_next):
    # 원하는 엔드포인트 경로를 확인하여 접근 제한 로직을 적용할 수 있습니다.
    if request.url.path in ["/docs", '/redoc']:
        # 접근 제한에 해당하는 경우 응답을 반환하고 요청 처리를 종료합니다.
        return JSONResponse(status_code=403, content={"detail": "Access denied"})

    # 다음 핸들러로 요청을 전달합니다.
    response = await call_next(request)
    return response

for router in routers:
    exec(f"app.include_router({router})")
    app.mount(f"/{router}", StaticFiles(directory=f"static/{router}",
              html=True), name=router)

app.mount("/", StaticFiles(directory="static", html=True), name="index")

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.0.14", port=2030)
