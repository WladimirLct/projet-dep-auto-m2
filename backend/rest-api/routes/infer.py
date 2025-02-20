from fastapi import File, UploadFile, HTTPException
from fastapi import APIRouter
from config.schemas import ONNXConfig
import requests
import httpx

import os
inference_url = os.getenv("INFERENCE_API_SERVICE_NAME")

router = APIRouter(prefix="/v1")


class ONNXClient:
    def __init__(self, config):
        self.client = httpx.Client(verify=config.verify_ssl)
        self.url = config.url
    
    def _generate_request(self, file):
        return {
            "file": (file.filename, file.file, file.content_type)
        }
    
    def post(self, file):
        try:
            return requests.post(self.url, files=self._generate_request(file))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error forwarding file: {str(e)}")

config = ONNXConfig(url=f"http://{inference_url}/infer", verify_ssl=True)  # CORRECT
client = ONNXClient(config)

@router.post("/infer")
async def infer_image(file: UploadFile = File(...)):
    if file.filename == "":
        raise HTTPException(status_code=400, detail="No file selected")
    
    response = client.post(file)
    
    if response.status_code == 200:
        return response.json()
    raise HTTPException(
        status_code=response.status_code,
        detail=f"Inference API error: {response.text}"
    )