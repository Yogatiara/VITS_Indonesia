from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class HTTTPException(Exception):
    def __init__(self, detail: str):
        self.detail = detail


app = FastAPI()

@app.exception_handler(HTTTPException)
async def create_handler(request: Request, exc: HTTTPException):
    return JSONResponse(
        status_code= status.HTTP_404_NOT_FOUND,
        detail= exc.detail
    )