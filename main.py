from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

@app.on_event("startup")
def load_models():
    app.state.yolo_model = YOLO("yolo11n.pt")

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
    
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = app.state.yolo_model(image)

    detections = []
    result = results[0]

    for box in result.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        name = result.names[cls_id]

        x1, y1, x2, y2 = map(float, box.xyxy[0])

        detections.append(
            {
                "class_id": cls_id,
                "class_name": name,
                "confidence" : conf,
                "bbox": [x1, y1, x2, y2]
            }
        )
    
    return {
        "model": "yolo11n",
        "filename": file.filename,
        "content-type": file.content_type,
        "num_detections": len(detections),
        "detections": detections
    }