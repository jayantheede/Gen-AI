from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["remodel_catalog"]

collections = ["unified_nodes", "image_embeddings", "embeddings_v1"]

print(f"Checking Database: remodel_catalog")
for col_name in collections:
    if col_name in db.list_collection_names():
        count = db[col_name].count_documents({})
        print(f"[OK] {col_name}: {count} documents")
        if count > 0:
            sample = db[col_name].find_one()
            if sample:
                print(f"   Sample keys: {list(sample.keys())}")
    else:
        print(f"[MISSING] {col_name}: NOT FOUND")
