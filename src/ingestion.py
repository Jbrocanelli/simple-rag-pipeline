from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

LOADERS = [
      {"glob": "**/*.pdf", "loader_cls": PyMuPDFLoader, "loader_kwargs": {}, "file_type": "pdf"},
      {"glob": "**/*.txt", "loader_cls": TextLoader, "loader_kwargs": {"encoding": "utf-8"}, "file_type": "txt"},
  ]

def load_documents(directory: str):
    doc_dir = Path(directory)

    if not doc_dir.exists():
        raise ValueError(f"Directory not found: {directory}")
    
    all_docs = []

    for loader in LOADERS:
        documents = DirectoryLoader(
        doc_dir,
        glob=loader["glob"],
        loader_cls=loader["loader_cls"],
        loader_kwargs=loader["loader_kwargs"]
        ).load()

        for d in documents:
            d.metadata["file_name"] = Path(d.metadata["source"]).name 
            d.metadata["file_type"] = loader["file_type"]

        all_docs.extend(documents)

    return all_docs


def split_documents(documents: list, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    return text_splitter.split_documents(documents)


    



    
