import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import FastAPI
import uvicorn

from . import models
from .responses_api import router as responses_router
from .session import router as session_router

app = FastAPI()
app.include_router(session_router)
app.include_router(responses_router)


@app.on_event("startup")
async def _startup() -> None:
    models.load_models()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
