from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

class TextRequest(BaseModel):
    prompt: str

@app.post("/llm/predict")
def llm_predict(body: TextRequest):
    return{
        "model": 'bitnet-stub',
        "input": body.prompt,
        "output": f"Echo: {body.prompt}"
    }

@app.post("/vision/detect")
async def vision_detect(file: UploadFile = File(...)):
    return {
        "model": "yolo-stub",
        "filename": file.filename,
        "content-type": file.content_type,
        "detections": []
    }