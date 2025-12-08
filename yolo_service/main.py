from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import io
from pymongo import MongoClient
import datetime
import os
import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth
import aio_pika
import asyncio
import json
import httpx
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.5"))
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY")

class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str

app = FastAPI()

request_counter = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["service", "path", "method", "status_code"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    response = await call_next(request)
    request_counter.labels(
        service = "yolo_service",
        path = request.url.path,
        method = request.method,
        status_code = str(response.status_code),
    ).inc()
    return response

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

rabbitmq_connection = None
rabbitmq_channel = None

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
    global rabbitmq_connection, rabbitmq_channel

    rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
    max_retries = 10

    for attempt in range(1, max_retries + 1):
        try:
            print(f"YOLO: connecting to RabbitMQ at {rabbitmq_url} (attempt {attempt})")
            rabbitmq_connection = await aio_pika.connect_robust(rabbitmq_url)
            rabbitmq_channel = await rabbitmq_connection.channel()

            queue_name = os.getenv("RABBITMQ_QUEUE", "bitnet_yolo_queue")
            await rabbitmq_channel.declare_queue(queue_name, durable = True)
            print("YOLO: declared queue", queue_name)

            print("YOLO: connected to RabbitMQ")
            break
        except Exception as e:
            print("YOLO: RabbitMQ connection failed:", e)
            if attempt == max_retries:
                print("YOLO: giving up on RabbitMQ connection after max retires")
                rabbitmq_connection = None
                rabbitmq_channel = None
                break
        
            await asyncio.sleep(3)

async def publish_message(message: dict):
    global rabbitmq_channel
    if rabbitmq_channel is None:
        print("YOLO: publish_message called but RabbitMQ is not connected")
        return
    
    body = json.dumps(message).encode()

    try:
        queue_name = os.getenv("RABBITMQ_QUEUE", "bitnet_yolo_queue")
        exchange = rabbitmq_channel.default_exchange
        print(f"YOLO: about to publish to {queue_name} via default exchange")
        await exchange.publish(
            aio_pika.Message(body = body),
            routing_key = queue_name,
            mandatory = True
        )
        print(f"YOLO: published message to {queue_name}", message)
    except Exception as e:
        print("YOLO: ERROR publishing RabbitMQ message:", repr(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type = CONTENT_TYPE_LATEST)

@app.post("/auth/register")
def register_user(req: RegisterRequest):
    try:
        user_record = firebase_auth.create_user(
            email = req.email,
            password = req.password
        )
        return {
            "uid": user_record.uid,
            "email": user_record.email
        }
    except Exception:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "Failed to create user"
        )


@app.post("/auth/login")
async def login_user(req: LoginRequest):
    if not FIREBASE_API_KEY:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = "FIREBASE_API_KEY not configured"
        )

    url = (
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
        f"?key={FIREBASE_API_KEY}"
    )
    payload = {
        "email": req.email,
        "password": req.password,
        "returnSecureToken": True
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json = payload)

    if resp.status_code != 200:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "Invalid email or password"
        )

    data = resp.json()
    return {
        "id_token": data["idToken"],
        "refresh_token": data.get("refreshToken"),
        "expires_in": data.get("expiresIn"),
        "user_id": data.get("localId")
    }


@app.post("/vision/detect")
async def vision_detect(file: UploadFile = File(...), user: dict = Depends(verify_firebase_token)):
    
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = app.state.yolo_model(image)

    detections = []
    result = results[0]

    for box in result.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        if conf < CONF_THRESHOLD:
            continue
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
        "created_at": datetime.datetime.utcnow(),
        "user_id": user.get("uid")
    }

    insert_result = app.state.detections.insert_one(doc)
    detection_id = str(insert_result.inserted_id)

    firebase_doc = {k: v for k, v in doc.items() if k != "_id"}
    firebase_ref = app.state.firestore.collection("yolo_detections").add(firebase_doc)
    firebase_doc_id = firebase_ref[1].id

    object_names = [det["class_name"] for det in detections]

    message = {
        "doc_id": detection_id,
        "objects": object_names,
        "source": "yolo_service"
    }

    await publish_message(message)

    return {
        "model": "yolo11n",
        "filename": file.filename,
        "content_type": file.content_type,
        "detections": detections,
        "num_detections": len(detections),
        "doc_id": detection_id,
        "firebase_doc_id": firebase_doc_id
    }

@app.get("/detections")
def list_detections(limit: int = 10, user: dict = Depends(verify_firebase_token)):
    cursor = (
        app.state.detections
        .find({"user_id": user.get("uid")})
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

@app.post("/test/rabbit")
async def test_rabbitmq():
    message = {
        "status" : "ok",
        "source" : "yolo_service",
        "message" : "RabbitMQ connectivity test"
     }
    
    await publish_message(message)

    return {"sent" : True}
