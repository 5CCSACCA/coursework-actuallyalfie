from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from pymongo import MongoClient
import datetime
import torch
import os
import firebase_admin
from firebase_admin import credentials, firestore


app = FastAPI()
MODEL_NAME = "microsoft/bitnet-b1.58-2B-4T"

@app.on_event("startup")
def load_model():
    app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    app.state.model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype = torch.float32,
        device_map = "cpu"
    )

    mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017")
    client = MongoClient(mongo_uri)
    app.state.db = client["coursework_db"]
    app.state.completions = app.state.db["completions"]

    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
    app.state.firestore = firestore.client()

@app.get("/health")
def health():
    return {"status": "ok"}

class TextRequest(BaseModel):
    prompt: str

@app.post("/llm/predict")
def predict(req: TextRequest):
    tokenizer = app.state.tokenizer
    model = app.state.model
    
    inputs = tokenizer(req.prompt, return_tensors = "pt")
    outputs = model.generate(**inputs, max_new_tokens = 50)

    result = tokenizer.decode(outputs[0], skip_special_tokens = True)

    document = {
        "model": MODEL_NAME,
        "prompt": req.prompt,
        "output": result,
        "created_at": datetime.datetime.utcnow()
    }
    app.state.completions.insert_one(document)
    firebase_doc = {k: v for k, v in document.items() if k != "_id"}
    app.state.firestore.collection("llm_completions").add(firebase_doc)

    return {"model": MODEL_NAME, "prompt": req.prompt, "output": result}

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