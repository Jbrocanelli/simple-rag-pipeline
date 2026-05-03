from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

def create_retrieval_chain(vectorstore, model_name: str = "llama-3.3-70b-versatile"):
    load_dotenv()

    llm = ChatGroq(model_name=model_name, temperature=0.1)

    prompt = PromptTemplate(
        input_variables=["context", "question"], 
        template="""Use the context below to answer the question. If you don't know, say so.

        Context: {context}

        Question: {question}

        Answer:"""
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3}
    )

    # the | operator pipes the output of each step as input to the next
    # RunnablePassThrough passes the unchanged original query as "question"
    # eg. "context": chunks, "question": "what is attention"

    # prompt takes the dict and fills the prompt template placeholders

    # llm sends the formatted prompt to grok and gets back a ChatMessage object(not a string)

    # StrOutputParser extracts the text content from the ChatMessage object and returns a plain string
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
