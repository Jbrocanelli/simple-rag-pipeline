from src.ingestion import load_documents, split_documents
from src.vectorstore import get_or_create_vector_db
from src.retriever import create_retrieval_chain

PERSIST_DIR = "data/vectorstore"

def main():
    documents = load_documents("data")
    split_docs = split_documents(documents)
    vector_db = get_or_create_vector_db(split_docs, persist_directory=PERSIST_DIR)
    chain = create_retrieval_chain(vectorstore=vector_db)

    while True:
        question = input("\nAsk a question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break
        print(chain.invoke(question))
       

if __name__ == "__main__":
    main()