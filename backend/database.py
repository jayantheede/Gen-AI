from pymongo import MongoClient
import os
from dotenv import load_dotenv
import pymongo
import re

load_dotenv()

class DatabaseHandler:
    def __init__(self, uri: str = None, db_name: str = "remodel_catalog"):
        self.uri = uri or os.getenv("MONGO_URI")
        print(f"Connecting to MongoDB with URI: {self.uri}")
        self.client = MongoClient(self.uri)
        self.db = self.client[db_name]
        self.embeddings = self.db.embeddings_v1
        self.image_embeddings = self.db.image_embeddings
        self.unified_collection = self.db.unified_nodes
        self._broken_filters = {"category"} # Known broken in current Atlas config

    def _execute_resilient_search(self, collection, search_params, filter_dict=None):
        """
        Executes a $vectorSearch with a fallback to manual filtering if the index
        is not configured for filtering on the requested path.
        """
        full_params = search_params.copy()
        if filter_dict:
            full_params["filter"] = filter_dict
            
        pipeline = [{"$vectorSearch": full_params}]
        
        # Optimization: Skip if we know this filter is broken
        if filter_dict:
            for k in filter_dict.keys():
                if k in self._broken_filters:
                    print(f"   |_ [INFO] Skipping known broken filter path: {k}")
                    return self._manual_fallback(collection, search_params, filter_dict)

        try:
            return list(collection.aggregate(pipeline))
        except pymongo.errors.OperationFailure as e:
            if "indexed as filter" in str(e) and filter_dict:
                for k in filter_dict.keys(): self._broken_filters.add(k)
                print(f"\n[!!!] ATLAS INDEX ERROR: {e}")
                return self._manual_fallback(collection, search_params, filter_dict)
            else:
                raise e

    def _manual_fallback(self, collection, search_params, filter_dict):
        print(f"   |_ [FALLBACK] Manual filtering for: {filter_dict}")
        retry_params = search_params.copy()
        # Higher limits for fallback to ensure we find matches after manual filtering
        retry_params["limit"] = 100 
        retry_params["numCandidates"] = 250 
        if "filter" in retry_params: del retry_params["filter"]
        
        retry_pipeline = [{"$vectorSearch": retry_params}]
        results = list(collection.aggregate(retry_pipeline))
        
        filtered = []
        for doc in results:
            match = True
            for k, v in filter_dict.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                filtered.append(doc)
        
        return filtered[:search_params.get("limit", 5)]

    def vector_search(self, query_embedding, limit=5, filter_dict=None):
        search_params = {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 100,
            "limit": limit
        }
        return self._execute_resilient_search(self.embeddings, search_params, filter_dict)

    def get_images_by_link_id(self, link_id):
        return list(self.image_embeddings.find({"link_id": link_id}))

    def visual_search(self, clip_embedding, limit=5, filter_dict=None):
        search_params = {
            "index": "image_vector_index",
            "path": "embedding",
            "queryVector": clip_embedding,
            "numCandidates": 100,
            "limit": limit
        }
        return self._execute_resilient_search(self.image_embeddings, search_params, filter_dict)

    def unified_search(self, query_embedding, limit=5, filter_dict=None):
        search_params = {
            "index": "vector_index", 
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 100,
            "limit": limit
        }
        return self._execute_resilient_search(self.unified_collection, search_params, filter_dict)

    def keyword_search(self, term, limit=5, category=None):
        """
        Direct text search for Article Numbers or specific keywords.
        """
        query = {"combined_text": {"$regex": re.escape(term), "$options": "i"}}
        if category:
            query["category"] = category
            
        return list(self.unified_collection.find(query).limit(limit))

    def strict_visual_search(self, clip_text_embedding, category, limit=5):
        """
        Search with hard category filtering at the vector index level.
        """
        search_params = {
            "index": "unified_clip_index",
            "path": "related_images.clip_embedding",
            "queryVector": clip_text_embedding,
            "numCandidates": 100,
            "limit": limit
        }
        filter_dict = {"category": category} if category else None
        return self._execute_resilient_search(self.unified_collection, search_params, filter_dict)
