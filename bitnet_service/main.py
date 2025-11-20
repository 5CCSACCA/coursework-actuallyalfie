from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

@app.on_event("startup")
def load_model():
    model_name = "microsoft/bitnet-b1.58-2B-4T"

    app.state.tokenizer = AutoTokenizer.from_pretrained(model_name)
    app.state.model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.float32,
    )

class TextRequest(BaseModel):
    prompt: str

@app.post("/llm/predict")
def predict(req: TextRequest):
    tokenizer = app.state.tokenizer
    model = app.state.model
    
    inputs = tokenizer(req.prompt, return_tensors = "pt")
    outputs = model.generate(**inputs, max_new_tokens = 50)

    result = tokenizer.decode(outputs[0], skip_special_tokens = True)
    return {"input": req.prompt, "output": result}