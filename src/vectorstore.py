from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path

def get_or_create_vector_db(documents: list, persist_directory: str, model_name: str = "all-MiniLM-L6-v2"):
      embedding = HuggingFaceEmbeddings(model_name=model_name)
      if Path(persist_directory).exists():
            return Chroma(persist_directory=persist_directory, embedding_function=embedding)

      return Chroma.from_documents(
          documents=documents,
          embedding=embedding,
          persist_directory=persist_directory,
      )
