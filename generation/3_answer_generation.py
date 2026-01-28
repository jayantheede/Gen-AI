import os
import ssl
import certifi
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Configuration
GROQ_API_KEY = "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"
MONGO_URI = "mongodb+srv://managerasasolutions_db_user:23092024@cluster0.m7lpbjq.mongodb.net/?appName=Cluster0"
DB_NAME = "rag_db"
COLLECTION_NAME = "embeddings"
INDEX_NAME = "vector_index"

# Initialize embeddings and MongoDB client
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = MongoClient(
    MONGO_URI,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=5000,
    connectTimeoutMS=5000,
    retryWrites=True
)
collection = client[DB_NAME][COLLECTION_NAME]

# Initialize Vector Store
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding_model,
    index_name=INDEX_NAME
)

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)

# Define retrieval prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Set up the RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vector_store.as_retriever(), question_answer_chain)

# Interactive Chat Loop
print("=== Interactive RAG Chat (MongoDB & Groq) ===")
print("Type 'exit' to quit.\n")

while True:
    try:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        if not query.strip():
            continue

        print("\nRetrieving context and generating answer...")
        result = rag_chain.invoke({"input": query})

        print("\n--- AI Response ---")
        print(result["answer"])
        print("\n" + "-"*50 + "\n")
    except EOFError:
        break
    except Exception as e:
        print(f"Error: {e}")
