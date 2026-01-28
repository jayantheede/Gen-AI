import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = "mongodb+srv://managerasasolutions_db_user:23092024@cluster0.m7lpbjq.mongodb.net/?appName=Cluster0"
DB_NAME = "rag_db"
COLLECTION_NAME = "embeddings"
INDEX_NAME = "vector_index"  # Ensure this index is created in MongoDB Atlas

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        # Create a sample file if it doesn't exist to avoid error
        with open(os.path.join(docs_path, "sample.txt"), "w") as f:
            f.write("Microsoft acquired GitHub for $7.5 billion in 2018.")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    """Create and store in MongoDB Atlas Vector Search"""
    print("Creating embeddings and storing in MongoDB Atlas...")
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    
    # Clear existing data if any (optional)
    # collection.delete_many({})
    
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection=collection,
        index_name=INDEX_NAME
    )
    
    print(f"Documents stored in MongoDB collection: {COLLECTION_NAME}")
    return vector_store

def main():
    print("=== RAG Document Ingestion Pipeline (MongoDB) ===\n")
    
    # Step 1: Load documents
    documents = load_documents("docs")  

    # Step 2: Split into chunks
    chunks = split_documents(documents)
    
    # Step 3: Create vector store in MongoDB
    create_vector_store(chunks)
    
    print("\n✅ Ingestion complete! Your documents are now ready in MongoDB Atlas.")

if __name__ == "__main__":
    main()
