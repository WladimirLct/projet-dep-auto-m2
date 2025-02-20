from fastapi import FastAPI, status
from routes import infer
import uvicorn

app = FastAPI(title="YOLO REST API Gateway")

# Update this URL to match your inference API endpoint

app.include_router(infer.router)

@app.get(
    "/healthcheck",
    tags=["healthcheck"],
    status_code=status.HTTP_200_OK,
    response_description="ok",
    summary="resume",
)
def get_api_status() -> str:
    """Return Ok if the api is up."""
    return "ok"