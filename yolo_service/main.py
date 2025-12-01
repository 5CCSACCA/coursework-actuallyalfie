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
import aio_pika
import asyncio
import json

app = FastAPI()

rabbitmq_connection = None
rabbitmq_channel = None
rabbitmq_exchange = None

@app.on_event("startup")
def startup_database():
    app.state.yolo_model = YOLO("yolo11n.pt")

    mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017")
    client = MongoClient(mongo_uri)
    app.state.db = client["coursework_db"]
    app.state.detections = app.state.db["detections"]

    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
    app.state.firestore = firestore.client()

@app.on_event("startup")
async def startup_rabbitmq():
    global rabbitmq_connection, rabbitmq_channel, rabbitmq_exchange

    rabbitmq_url = os.getenv("RABBITMQ_URL")

    rabbitmq_connection = await aio_pika.connect_robust(rabbitmq_url)
    rabbitmq_channel = await rabbitmq_connection.channel()
    rabbitmq_exchange = await rabbitmq_channel.declare_exchange(
        "yolo_exchange",
        aio_pika.ExchangeType.FANOUT
    )

async def publish_message(message: dict):
    body = json.dumps(message).encode()
    await rabbitmq_exchange.publish(
        aio_pika.Message(body = body),
        routing_key = ""
    )

@app.get("/health")
def health():
    return {"status": "ok"}


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
    firebase_doc = {k: v for k, v in doc.items() if k != "_id"}
    app.state.firestore.collection("yolo_detections").add(firebase_doc)

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

@app.get("/firebase/yolo/{doc_id}")
def get_yolo_detection(doc_id: str):
    doc = app.state.firestore.collection("yolo_detections").document(doc_id).get()
    return doc.to_dict() or {}

@app.put("/firebase/yolo/{doc_id}")
def update_yolo_detection(doc_id: str, updates: dict):
    app.state.firestore.collection("yolo_detections").document(doc_id).update(updates)
    return {"status": "updated"}

@app.delete("/firebase/yolo/{doc_id}")
def delete_yolo_detection(doc_id: str):
    app.state.firestore.collection("yolo_detections").document(doc_id).delete()
    return {"status": "deleted"}
