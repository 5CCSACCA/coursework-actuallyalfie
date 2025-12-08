from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
import datetime
import httpx
import os
import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth
import aio_pika
import asyncio
import json

app = FastAPI()

security_scheme = HTTPBearer(auto_error = False)

def verify_firebase_token(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Missing or invalid Authorization header"
        )
    id_token = credentials.credentials
    try:
        decoded_token = firebase_auth.verify_id_token(id_token)
        return decoded_token
    except Exception:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid or expired Firebase ID token"
        )

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

def save_llm_completion(prompt: str, result: str, extra: dict | None = None) -> dict:
    document = {
        "model": BITNET_MODEL_NAME,
        "prompt": prompt,
        "output": result,
        "created_at": datetime.datetime.utcnow()
    }
    if extra:
        document.update(extra)

    app.state.completions.insert_one(document)
    firebase_doc = {k: v for k, v in document.items() if k != "_id"}
    app.state.firestore.collection("llm_completions").add(firebase_doc)

    return document

@app.post("/llm/predict")
async def predict(req: TextRequest, user: dict = Depends(verify_firebase_token)):
    completion = await call_bitnet_llm(req.prompt)

    try:
        result = completion["choices"][0]["message"]["content"]
    except Exception:
        result = str(completion)

    save_llm_completion(req.prompt, result, extra = {"user_id": user.get("uid")})

    return {"model": BITNET_MODEL_NAME, "prompt": req.prompt, "output": result}

@app.get("/llm/completions")
def list_completions(limit: int = 10, user: dict = Depends(verify_firebase_token)):
    cursor = (
        app.state.completions
        .find({"user_id": user.get("uid")})
        .sort("created_at", -1)
        .limit(limit)
    )

    items = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        items.append(doc)

    return {"count": len(items), "items": items}

@app.get("/firebase/llm/{doc_id}")
def get_llm_completion(doc_id: str, user: dict = Depends(verify_firebase_token)):
    doc = app.state.firestore.collection("llm_completions").document(doc_id).get()
    return doc.to_dict() or {}

@app.put("/firebase/llm/{doc_id}")
def update_llm_completion(doc_id: str, updates: dict, user: dict = Depends(verify_firebase_token)):
    app.state.firestore.collection("llm_completions").document(doc_id).update(updates)
    return {"status": "updated"}

@app.delete("/firebase/llm/{doc_id}")
def delete_llm_completion(doc_id: str, user: dict = Depends(verify_firebase_token)):
    app.state.firestore.collection("llm_completions").document(doc_id).delete()
    return {"status": "deleted"}


    
async def on_message(message: aio_pika.IncomingMessage):
    async with message.process():
        try:
            body_text = message.body.decode()
            print("BitNet: decoded message body:", body_text)

            payload = json.loads(body_text)
            print("Received RabbitMQ message:", payload)

            doc_id = payload.get("doc_id")
            objects = payload.get("objects", [])

            if not doc_id:
                print("BitNet: message missing doc_id, skipping")
                return

            detections = app.state.db["detections"]
            detection_doc = None
            try:
                detection_doc = detections.find_one({"_id": ObjectId(doc_id)})
            except Exception as fetch_err:
                print("BitNet: failed to fetch detection:", repr(fetch_err))

            objects_str = ", ".join(objects)
            prompt_parts = [
                "You are helping a user understand what is in a photo.",
                f"The detected objects are: {objects_str or 'none'}."
            ]

            if detection_doc is not None:
                extra_meta = {k: v for k, v in detection_doc.items() if k != "_id"}
                prompt_parts.append(f"Additional detection metadata: {extra_meta}")

            prompt_parts.append(
                "Write 2-3 friendly sentences describing the scene, "
                "then provide one short caption for the image in quotes. "
                "Keep the answer under 80 words."
            )
            prompt = "\n".join(prompt_parts)

            completion = await call_bitnet_llm(prompt)
            try:
                result = completion["choices"][0]["message"]["content"]
            except Exception:
                result = str(completion)

            extra = {"detection_doc_id": doc_id, "objects": objects}
            save_llm_completion(prompt, result, extra)
            print("BitNet: LLM post-processing complete for detection", doc_id)
        except Exception as e:
            print("BitNet: error processing message:", repr(e))