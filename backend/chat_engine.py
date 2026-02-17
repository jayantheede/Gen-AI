import os
import re
import time
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .database import DatabaseHandler
from .rag_tools import RAGTools


class ChatEngine:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.db = DatabaseHandler()
        self.unified_collection = self.db.db["unified_nodes"]
        self.image_collection = self.db.db["image_embeddings"]
        self.rag_tools = RAGTools()

        self.system_prompt = (
            "You are an expert automotive consultant for Wurthi Automotive Solutions. "
            "Your knowledge is strictly limited to cars, vehicles, automotive engineering, and catalog specifications. "
            "STRICT RULE: If the user asks about anything unrelated to automotive design or specifications (such as politics, celebrities, sports, or general off-topic questions), "
            "you must politely refuse and state that you are only optimized for automotive inquiries. "
            "Never answer general knowledge questions outside the automotive domain. "
            "Always use the provided context to justify your automotive advice."
            "\n\nContext:\n{context}"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{question}"),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

        self.pdf_mapping = {
            "automotive": "/data/Automotive_Catalogue_4_April_2025.pdf"
        }

    # ============================================================
    # AUTO ROUTER ENTRY
    # ============================================================

    def ask(self, question: str, rag_mode="auto"):
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"QUERY: {question}")
        print(f"{'='*60}\n")

        initial_docs = None
        if rag_mode == "auto":
            print("[AUTO ROUTER] Analyzing query...")
            initial_docs = self._retrieve_context(question, top_k=5)
            wc = len(question.split())

            if wc <= 3:
                rag_mode = "speculative"
                print("[AUTO ROUTER] Short query → Speculative Mode")
            elif len(initial_docs) < 3:
                rag_mode = "fusion"
                print("[AUTO ROUTER] Low recall → Fusion Mode")
            else:
                rag_mode = "standard"
                print("[AUTO ROUTER] High confidence → Standard Mode")

        # Route to appropriate pipeline
        if rag_mode == "standard":
            result = self._standard_rag_pipeline(question, initial_docs)
        elif rag_mode == "corrective":
            result = self._corrective_rag_pipeline(question, initial_docs)
        elif rag_mode == "speculative":
            result = self._speculative_rag_pipeline(question, initial_docs)
        elif rag_mode == "fusion":
            result = self._fusion_rag_pipeline(question, initial_docs)
        else:
            print(f"Unknown mode '{rag_mode}', defaulting to standard")
            result = self._standard_rag_pipeline(question, initial_docs)
            rag_mode = "standard"

        total_time = time.time() - start_time
        print(f"\n[BENCHMARK] Total Pipeline Time: {total_time:.2f}s")
        result["mode"] = rag_mode
        result["generation_time"] = f"{total_time:.2f}s"
        return result

    # ============================================================
    # PIPELINES
    # ============================================================

    def _standard_rag_pipeline(self, question, initial_docs=None):
        print("[STANDARD] Starting pipeline...")
        category = self._detect_category(question)
        
        # Reuse initial_docs if available to save one full search cycle (20-70s)
        if initial_docs:
            print("   |_ [INFO] Using router-cached results")
            context_docs = initial_docs
        else:
            context_docs = self._retrieve_context(question, 10, category)
            
        context = "\n\n".join([d.get("combined_text","") for d in context_docs])
        
        # Standard search uses original question for images
        images = self._retrieve_images(question, category, 12, context_docs)
        answer = self._generate_answer(question, context)
        return {"answer": answer, "images": images}

    def _corrective_rag_pipeline(self, question, initial_docs=None):
        print("[CORRECTIVE] Starting pipeline...")
        category = self._detect_category(question)
        
        if initial_docs:
            context_docs = initial_docs
        else:
            context_docs = self._retrieve_context(question, 8, category)
            
        context = "\n\n".join([d.get("combined_text","") for d in context_docs])
        score = self._score_relevance(question, context)
        print(f"[CORRECTIVE] Relevance Score: {score:.2f}")

        if score < 0.7:
            print("[CORRECTIVE] Low relevance. Rewriting query for better catalog coverage...")
            rewritten_q = self._rewrite_query(question)
            print(f"[CORRECTIVE] New Query: {rewritten_q}")
            secondary = self._retrieve_context(rewritten_q, 15, None)
            context_docs = self._deduplicate_docs(context_docs + secondary)
            
            # Use REWRITTEN query for images in Corrective mode
            v_query = rewritten_q
        else:
            v_query = question

        context = "\n\n".join([d.get("combined_text","") for d in context_docs[:12]])
        images = self._retrieve_images(v_query, category, 12, context_docs)
        answer = self._generate_answer(question, context)

        return {"answer": answer, "images": images, "relevance_score": score}

    def _speculative_rag_pipeline(self, question, initial_docs=None):
        print("[SPECULATIVE] Starting pipeline...")
        category = self._detect_category(question)
        
        # Use initial docs as the 'quick' setup to save time
        if initial_docs:
            quick_docs = initial_docs[:3]
        else:
            quick_docs = self._retrieve_context(question, 3, category)
            
        quick_context = "\n\n".join([d.get("combined_text","") for d in quick_docs])
        
        # Step 1: Generate draft and catch entities
        draft = self._generate_draft_answer(question, quick_context)
        entities = self._extract_entities(f"{question} {draft}")
        # Reduce search width to keep it fast on CPU
        print(f"[SPECULATIVE] Entities for enrichment (Limiting to top 2): {entities[:2]}")

        # Step 2: Parallel enrichment (Limited)
        enriched = []
        for entity in entities[:2]: 
            enriched.extend(self._retrieve_context(entity, 3, None))
            
        # Step 3: Synthesis
        merged = self._deduplicate_docs(quick_docs + enriched)
        merged = self._rerank_by_relevance(question, merged, 15)
        
        context = "\n\n".join([d.get("combined_text","") for d in merged])
        answer = self._generate_answer(question, context)
        
        # Use extracted entities to drive image discovery
        v_query = f"{question} {' '.join(entities[:2])}"
        images = self._retrieve_images(v_query, category, 12, merged)

        return {"answer": answer, "images": images, "entities": entities}

    def _fusion_rag_pipeline(self, question, initial_docs=None):
        print("[FUSION] Starting pipeline...")
        category = self._detect_category(question)
        
        # Step 1: Generate query variations
        queries = self._generate_queries(question)
        # Limit variations on CPU
        queries = queries[:2]
        print(f"[FUSION] Variations (Limited for speed): {queries}")
        
        # Step 2: Multi-search
        all_results = []
        for q in queries:
            docs = self._retrieve_context(q, top_k=10, category=category)
            all_results.append(docs)
            
        # Step 3: Reciprocal Rank Fusion
        fused_docs = self._reciprocal_rank_fusion(all_results, limit=15)
        print(f"[FUSION] Merged into {len(fused_docs)} best matches via RRF")
        
        context = "\n\n".join([d.get("combined_text","") for d in fused_docs])
        images = self._retrieve_images(question, category, 12, fused_docs)
        answer = self._generate_answer(question, context)
        
        return {"answer": answer, "images": images, "queries": queries}

    # ============================================================
    # FUSION HELPERS
    # ============================================================

    def _generate_queries(self, question):
        prompt = f"""You are a search expert. Rewrite the query below into 3 distinct semantic variations to improve catalog retrieval.
Avoid just changing synonyms; think about different architectural aspects of the query.

Original Query: {question}

Provide exactly 3 variations, one per line, no numbering:"""
        try:
            res = self.llm.invoke(prompt).content.strip()
            variations = [v.strip() for v in res.split("\n") if v.strip()]
            return variations[:3] + [question] # Always include original
        except:
            return [question]

    def _reciprocal_rank_fusion(self, results_lists, limit=15, k=60):
        """
        Standard RRF algorithm to merge ranked lists.
        k is a constant that controls how much priority is given to documents at the top.
        """
        fused_scores = {} # doc_id -> score
        doc_store = {}   # doc_id -> doc_obj
        
        for results in results_lists:
            for rank, doc in enumerate(results):
                doc_id = str(doc.get("_id"))
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                
                # RRF Formula: score = sum( 1 / (rank + k) )
                fused_scores[doc_id] += 1.0 / (rank + k)
                doc_store[doc_id] = doc
        
        # Sort by score
        ranked_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_store[doc_id] for doc_id, score in ranked_ids[:limit]]

    # ============================================================
    # CORE HELPERS
    # ============================================================

    def _retrieve_context(self, question, top_k=15, category=None):
        t0 = time.time()
        print(f"   |_ [INFO] Retrieving context for: '{question[:40]}...'")
        
        t_emb_start = time.time()
        emb = self.rag_tools.get_embeddings(question)
        t_emb = time.time() - t_emb_start
        
        t_db_start = time.time()
        res = self.db.unified_search(emb, limit=top_k, filter_dict={"category": category} if category else None)
        t_db = time.time() - t_db_start
        
        print(f"   |_ [ST] Total context hunt: {time.time()-t0:.2f}s (Emb: {t_emb:.2f}s, DB: {t_db:.2f}s)")
        return res

    def _retrieve_images(self, search_query, category, limit=12, text_docs=None):
        t0 = time.time()
        # Use the specific search_query (rewritten or enriched) for CLIP
        query_clip_emb = self.rag_tools.get_clip_text_embedding(search_query)
        print(f"   |_ [CLIP] Image query: '{search_query[:50]}...'")
        print(f"   |_ [CLIP] Text encoding took: {time.time()-t0:.2f}s")
        
        all_candidate_images = [] # List of formatted image objects with scores
        seen_paths = set()

        # Step 1: Collect candidates from Visual-Direct Search
        # Increase search breadth
        visual_nodes = self.db.strict_visual_search(query_clip_emb, category, limit=20)
        for node in visual_nodes:
            if "related_images" in node:
                for img in node["related_images"]:
                    img_path = img.get("image_path") or img.get("path")
                    if img_path and img_path not in seen_paths:
                        # Score it
                        img_emb = img.get("clip_embedding")
                        if img_emb:
                            sim = np.dot(query_clip_emb, img_emb) / (np.linalg.norm(query_clip_emb) * np.linalg.norm(img_emb) + 1e-8)
                            
                            # Strict threshold for visual accuracy
                            if sim > 0.22:
                                img["score"] = float(sim)
                                all_candidate_images.append(self._format_image_path(img, parent_node=node))
                                seen_paths.add(img_path)

        # Step 2: Collect candidates from Text-Context Search (with BOOST)
        if text_docs:
            for doc in text_docs:
                if "related_images" in doc:
                    for img in doc["related_images"]:
                        img_path = img.get("image_path") or img.get("path")
                        if img_path and img_path not in seen_paths:
                            img_emb = img.get("clip_embedding")
                            if img_emb:
                                sim = np.dot(query_clip_emb, img_emb) / (np.linalg.norm(query_clip_emb) * np.linalg.norm(img_emb) + 1e-8)
                                
                                # Boost images that match BOTH context and visual
                                # This ensures the gallery matches the text answer
                                final_score = float(sim) * 1.15
                                
                                if final_score > 0.22:
                                    img["score"] = final_score
                                    all_candidate_images.append(self._format_image_path(img, parent_node=doc))
                                    seen_paths.add(img_path)

        # Step 3: Global sort and filter
        all_candidate_images.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Performance logging
        if all_candidate_images:
            print(f"   |_ [CLIP] Top visual match score: {all_candidate_images[0]['score']:.4f}")

        return all_candidate_images[:limit]

    def _score_relevance(self, query, context):
        if not context.strip(): return 0.0
        prompt = f"""Rate the relevance of the following context to the user's question.
Question: {query}
Context: {context}

Rate from 0.0 to 1.0 (float only, e.g., 0.85):"""
        try:
            res = self.llm.invoke(prompt).content.strip()
            scores = re.findall(r"0\.\d+|1\.0", res)
            return float(scores[0]) if scores else 0.5
        except:
            return 0.5

    def _rewrite_query(self, query):
        prompt = f"""Rewrite the following user design query into a more technical, catalog-friendly search term.
Focus on architectural keywords, materials, and specific products.

Original: {query}
Technical Search Term:"""
        try:
            return self.llm.invoke(prompt).content.strip().replace('"', '')
        except:
            return query

    def _extract_entities(self, text):
        prompt = f"Extract exactly 5 key architectural or design entities (e.g., 'marble island', 'pendant lighting') from this text. Separated by commas:\n\n{text}"
        try:
            res = self.llm.invoke(prompt).content.strip()
            return [e.strip() for e in res.split(",") if e.strip()]
        except:
            return []

    def _detect_category(self, question):
        q = question.lower()
        if any(w in q for w in ["car", "vehicle", "engine", "speed", "safety", "luxury", "suv", "sedan", "wrench", "soldering", "pneumatic", "wiring", "electrical"]): return "automotive"
        return None

    def _deduplicate_docs(self, docs):
        seen = set()
        out = []
        for d in docs:
            did = str(d.get("_id"))
            if did not in seen:
                out.append(d)
                seen.add(did)
        return out

    def _rerank_by_relevance(self, query, docs, top_k=10):
        q_emb = np.array(self.rag_tools.get_embeddings(query))
        scored = []
        for d in docs:
            d_emb = np.array(d.get("embedding", []))
            if len(d_emb) == 0: continue
            score = np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-8)
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scored[:top_k]]

    def _format_image_path(self, img_obj, parent_node=None):
        path = img_obj.get("image_path") or img_obj.get("path", "")
        
        page = img_obj.get("page_label") or img_obj.get("page")
        category = "Catalog"
        
        if parent_node:
            page = page or parent_node.get("page")
            category = parent_node.get("category", "Catalog")
            
        pdf_url = self.pdf_mapping.get(category.lower(), "")
        if pdf_url and page:
            # Add page anchor for supported viewers (like Chrome/Edge internal PDF viewer)
            pdf_url = f"http://localhost:8000{pdf_url}#page={page}"

        return {
            "image_path": path,
            "page": page or "N/A",
            "pdf": category.title(),
            "pdf_url": pdf_url, 
            "ocr_text": img_obj.get("ocr_text", ""),
            "score": img_obj.get("score", 0.3)
        }

    def _generate_answer(self, question, context):
        try:
            res = self.chain.invoke({"context": context, "question": question})
            return res if res.strip() else "I'm sorry, I couldn't find enough information in our catalogs to answer that specific design question."
        except Exception as e:
            return f"I encountered an error while analyzing the catalogs: {e}"

    def _generate_draft_answer(self, question, context):
        prompt = f"Based on this partial context, provide a one-sentence design hypothesis:\nContext: {context}\nQuestion: {question}"
        try:
            return self.llm.invoke(prompt).content
        except:
            return "Seeking further catalog details..."
