import json, os, uvicorn
from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer

# embed_model = SentenceTransformer('sentence-transformers/sentence-t5-xl')
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

app = FastAPI()

@app.post("/embed")
async def embed_function(request: Request):
    body = await request.body()
    
    body = json.loads(body.decode("utf-8"))
    if isinstance(body["text"], list):
        embeddings = embed_model.encode(body["text"])
    else:
        embeddings = embed_model.encode([body["text"]])
    return {"array":embeddings.flatten().tolist(), "shape":embeddings.shape}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)