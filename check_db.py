from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
for db_name in client.list_database_names():
    db = client[db_name]
    cols = db.list_collection_names()
    print(f"DB: {db_name}")
    for col in cols:
        count = db[col].count_documents({})
        print(f"  - {col}: {count} docs")
