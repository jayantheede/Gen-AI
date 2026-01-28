import os
from langchain_mongodb import MongoDBAtlasVectorSearch, MongoDBChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Configuration
GROQ_API_KEY = "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"
MONGO_URI = "mongodb+srv://managerasasolutions_db_user:23092024@cluster0.m7lpbjq.mongodb.net/?appName=Cluster0"
DB_NAME = "rag_db"
COLLECTION_NAME = "embeddings"
INDEX_NAME = "vector_index"
SESSION_ID = "user_123"

# Initialize embeddings and MongoDB client
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# Initialize Vector Store
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding_model,
    index_name=INDEX_NAME
)

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)

# 1. Setup Chat History
chat_history = MongoDBChatMessageHistory(
    connection_string=MONGO_URI,
    session_id=SESSION_ID,
    database_name="chat_history_db",
    collection_name="messages"
)

# 2. Contextualize Question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, vector_store.as_retriever(), contextualize_q_prompt
)

# 3. Answer Question
qa_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def chat_step(query):
    # Get relevant history
    history = chat_history.messages
    
    # Run the chain
    result = rag_chain.invoke({"input": query, "chat_history": history})
    
    # Save the interaction to history
    chat_history.add_user_message(query)
    chat_history.add_ai_message(result["answer"])
    
    print("\n--- AI Response ---")
    print(result["answer"])

if __name__ == "__main__":
    print("=== Interactive History-Aware RAG (MongoDB & Groq) ===")
    print("Type 'exit' to quit.\n")
    
    # chat_history.clear() # Uncomment to reset history
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue
                
            chat_step(user_input)
            print("\n" + "-"*50 + "\n")
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")