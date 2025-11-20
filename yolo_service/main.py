from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import io
from pymongo import MongoClient
import datetime
import os
import firebase_admin
from firebase_admin import credentials, firestore

app = FastAPI()

@app.on_event("startup")
def startup_event():
    app.state.yolo_model = YOLO("yolo11n.pt")

    mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017")
    client = MongoClient(mongo_uri)
    app.state.db = client["coursework_db"]
    app.state.detections = app.state.db["detections"]

    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
    app.state.firestore = firestore.client()


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
    
    doc =  {
        "model": "yolo11n",
        "filename": file.filename,
        "content_type": file.content_type,
        "detections": detections,
        "created_at": datetime.datetime.utcnow()
    }

    app.state.detections.insert_one(doc)

    return {
        "model": "yolo11n",
        "filename": file.filename,
        "content_type": file.content_type,
        "detections": detections,
        "num_detections": len(detections)
    }

@app.get("/detections")
def list_detections(limit: int = 10):
    cursor = (
        app.state.detections
        .find({})
        .sort("created_at", -1)
        .limit(limit)
    )
    items = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        items.append(doc)
    return {"count": len(items), "items": items}

@app.post("/firebase/items")
def create_item(item: dict):
    doc_ref = app.state.firestore.collection("items").add(item)
    return {"id": doc_ref[1].id}

@app.get("/firebase/items/{item_id}")
def get_item(item_id: str):
    doc = app.state.firestore.collection("items").document(item_id).get()
    return doc.to_dict() or {}

@app.put("/firebase/items/{item_id}")
def update_item(item_id: str, updates: dict):
    app.state.firestore.collection("items").document(item_id).update(updates)
    return {"status": "updates"}

@app.delete("/firebase/items/{item_id}")
def delete_item(item_id: str):
    app.state.firestore.collection("items").document(item_id).delete()
    return {"status": "deleted"}