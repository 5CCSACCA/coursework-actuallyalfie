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

@app.on_event("startup")
async def startup():
    print("BitNet: startup() called")

    mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017")
    client = MongoClient(mongo_uri)
    app.state.db = client["coursework_db"]
    app.state.completions = app.state.db["completions"]

    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
    app.state.firestore = firestore.client()

    app.state.rabbitmq_task = asyncio.create_task(rabbitmq_worker())
    print("BitNet: RabbitMQ worker task created")

@app.on_event("shutdown")
async def shutdown():
    print("BitNet: shutdown() called")
    rabbitmq_task = getattr(app.state, "rabbitmq_task", None)
    if rabbitmq_task is not None:
        rabbitmq_task.cancel()
        try:
            await rabbitmq_task
        except asyncio.CancelledError:
            pass

async def rabbitmq_worker():
    rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
    queue_name = os.getenv("RABBITMQ_QUEUE", "bitnet_yolo_queue")

    while True:
        connection = None
        try:
            print("BitNet: connecting to RabbitMQ at", rabbitmq_url)
            connection = await aio_pika.connect_robust(rabbitmq_url)
            print("BitNet: connected to RabbitMQ")

            channel = await connection.channel()
            await channel.set_qos(prefetch_count = 1)

            queue = await channel.declare_queue(
                queue_name,
                durable = True
            )
            print("BitNet: declared queue", queue.name)

            await queue.consume(on_message)
            print("BitNet: worker listening for YOLO messages...")

            await asyncio.Future()
        except asyncio.CancelledError:
            print("BitNet: RabbitMQ worker cancelled")
            if connection is not None:
                await connection.close()
            break
        except Exception as e:
            print("BitNet: RabbitMQ worker error, retrying in 5 seconds:", repr(e))
            if connection is not None:
                try:
                    await connection.close()
                except Exception:
                    pass
            await asyncio.sleep(5)

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
            app.state.db["rabbitmq_debug"].insert_one(
                {
                    "recieved_at": datetime.datetime.utcnow(),
                    "raw_body": message.body.decode(errors = "replace")
                }
            )

            body_text = message.body.decode()
            print("BitNet: decoded message body:", body_text)

            payload = json.loads(body_text)
            print("Recieved RabbitMQ message:", payload)


        except Exception as e:
            print("BitNet: error processing message:", repr(e))