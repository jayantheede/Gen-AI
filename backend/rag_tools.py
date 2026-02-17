import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel


class RAGTools:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device detected: {self.device}")

        print("Loading Text Embedding Model (MiniLM)...")
        # Direct fix for Python 3.13: Disable low_cpu_mem_usage to prevent
        # the model from being loaded into meta-tensors initially.
        self.text_model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=self.device,
            model_kwargs={"low_cpu_mem_usage": False}
        )

        print("Loading CLIP Model...")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            low_cpu_mem_usage=False
        ).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        print("Models loaded.")
        self._clip_text_cache = {}
        self._text_cache = {}

    # ---------------- Text Embeddings ----------------
    @torch.inference_mode()
    def get_embeddings(self, text: str):
        if text in self._text_cache:
            return self._text_cache[text]
        result = self.text_model.encode(text).tolist()
        self._text_cache[text] = result
        return result


    # ---------------- Chunking ----------------

    def get_chunks(self, text, chunk_size=500, overlap=100):

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks


    # ---------------- CLIP Image Embeddings ----------------

    @torch.inference_mode()
    def get_clip_image_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        emb = self.clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0].tolist()


    # ---------------- CLIP Text Embedding (Optional) ----------------

    @torch.inference_mode()
    def get_clip_text_embedding(self, text):
        if text in self._clip_text_cache:
            return self._clip_text_cache[text]
            
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        emb = self.clip_model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        
        result = emb.cpu().numpy()[0].tolist()
        self._clip_text_cache[text] = result
        return result
