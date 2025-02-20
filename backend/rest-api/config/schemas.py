from pydantic import BaseModel, HttpUrl


class ONNXConfig(BaseModel):
    url: HttpUrl
    verify_ssl: bool = True
