from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.controller.inference_controller import router as inference_controller


app = FastAPI()

app.include_router(inference_controller, tags=["Inference"])
app.mount("/audio", StaticFiles(directory="source_audio"), name="audio")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()