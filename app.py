from fastapi import FastAPI, Depends
from pydantic import BaseModel
from functools import lru_cache
from src.ingestion import load_documents, split_documents
from src.vectorstore import get_or_create_vector_db
from src.retriever import create_retrieval_chain

PERSIST_DIR = "data/vectorstore"

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str # {"prompt": "..."}

class ChatResponse(BaseModel):
    response: str #{"response": "..."}

@lru_cache
# runs once then it's cached, lru means last recently used
def build_chain():
    documents = load_documents("data")
    split_docs = split_documents(documents)
    vector_db = get_or_create_vector_db(split_docs, persist_directory=PERSIST_DIR)
    return create_retrieval_chain(vectorstore=vector_db)



@app.get("/")
async def root():
    return {"message": "API is running"}

# injects build_chain as a dependency
@app.post("/ask", response_model=ChatResponse)
async def ask(request: ChatRequest, chain=Depends(build_chain)):
    answer = chain.invoke(request.prompt)
    return ChatResponse(response=answer)

   