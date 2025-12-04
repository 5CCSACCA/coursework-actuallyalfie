from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
import datetime
import httpx
import os
import firebase_admin
from firebase_admin import credentials, firestore
import aio_pika
import asyncio
import json

app = FastAPI()
BITNET_MODEL_NAME = "ggml-model-i2_s.gguf"
BITNET_URL = os.getenv("BITNET_URL", "http://bitnet:8080")

rabbitmq_connection = None
rabbitmq_channel = None
rabbitmq_queue = None

@app.on_event("startup")
async def load_model():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017")
    client = MongoClient(mongo_uri)
    app.state.db = client["coursework_db"]
    app.state.completions = app.state.db["completions"]

    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
    app.state.firestore = firestore.client()

@app.on_event("startup")
async def startup_rabbitmq():
    print("BitNet: startup_rabbitmq called")
    global rabbitmq_connection, rabbitmq_channel, rabbitmq_queue

    rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
    print("BitNet: connecting to RabbitMQ at", rabbitmq_url)

    while True:
        try:
            rabbitmq_connection = await aio_pika.connect_robust(rabbitmq_url)
            print("BitNet: connected to RabbitMQ")
            break
        except Exception as e:
            print("BitNet: waiting for RabbitMQ...", e)
            await asyncio.sleep(2)

    rabbitmq_channel = await rabbitmq_connection.channel()
    await rabbitmq_channel.set_qos(prefetch_count = 1)

    queue_name = "bitnet_yolo_queue"
    rabbitmq_queue = await rabbitmq_channel.declare_queue(
        queue_name,
        durable = True
    )
    print("BitNet: declared queue", rabbitmq_queue.name)

    await rabbitmq_queue.consume(on_message)
    print("BitNet: worker listening for YOLO messages...")

@app.get("/health")
def health():
    return {"status": "ok"}

class TextRequest(BaseModel):
    prompt: str

async def call_bitnet_llm(user_message: str) -> dict:
    payload = {
        "model": BITNET_MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a chatbot that answers questions. Don't overthink things."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    }

    async with httpx.AsyncClient(timeout = None) as client:
        response = await client.post(f"{BITNET_URL}/v1/chat/completions", json = payload)
        response.raise_for_status()
        return response.json()

@app.post("/llm/predict")
async def predict(req: TextRequest):
    completion = await call_bitnet_llm(req.prompt)

    try:
        result = completion["choices"][0]["message"]["content"]
    except Exception:
        result = str(completion)

    document = {
        "model": BITNET_MODEL_NAME,
        "prompt": req.prompt,
        "output": result,
        "created_at": datetime.datetime.utcnow()
    }
    app.state.completions.insert_one(document)
    firebase_doc = {k: v for k, v in document.items() if k != "_id"}
    app.state.firestore.collection("llm_completions").add(firebase_doc)

    return {"model": BITNET_MODEL_NAME, "prompt": req.prompt, "output": result}

@app.get("/llm/completions")
def list_completions(limit: int = 10):
    cursor = (
        app.state.completions
        .find({})
        .sort("created_at", -1)
        .limit(limit)
    )

    items = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        items.append(doc)

    return {"count": len(items), "items": items}

@app.get("/firebase/llm/{doc_id}")
def get_llm_completion(doc_id: str):
    doc = app.state.firestore.collection("llm_completions").document(doc_id).get()
    return doc.to_dict() or {}

@app.put("/firebase/llm/{doc_id}")
def update_llm_completion(doc_id: str, updates: dict):
    app.state.firestore.collection("llm_completions").document(doc_id).update(updates)
    return {"status": "updated"}

@app.delete("/firebase/llm/{doc_id}")
def delete_llm_completion(doc_id: str):
    app.state.firestore.collection("llm_completions").document(doc_id).delete()
    return {"status": "deleted"}


    
async def on_message(message: aio_pika.IncomingMessage):
    print("BitNet: on_message callback TRIGGERED")
    async with message.process():
        try:
            body_text = message.body.decode()
            print("BitNet: decoded message body:", body_text)

            payload = json.loads(body_text)
            print("Recieved RabbitMQ message:", payload)


        except Exception as e:
            print("BitNet: error processing message:", repr(e))