# Remodel Design Ingestion UI

A high-performance full-stack system to ingest remodeling and interior design catalogs into a MongoDB vector store with OCR and image extraction.

## Features
- **FastAPI Backend**: Asynchronous processing with background tasks.
- **Multimodal Extraction**: Native text, Tesseract OCR for scans, and PyMuPDF for images.
- **Admin Console**: [NEW] Specialized portal for data ingestion and human-in-the-loop OCR validation.
- **AI Chat Assistant**: [NEW] ChatGPT-style interface with RAG using Ollama (Llama 3).
- **Persistent Chat History**: Previous conversations are saved in MongoDB for seamless continuity.
- **RAG Ready**: Recursive character chunking and HuggingFace embeddings (`all-MiniLM-L6-v2`).
- **Modern UI**: Simple, glassmorphism UI with Tailwind CSS.
- **Progress Tracking**: Real-time polling for page processing and extraction status.

## Setup

### 1. Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com/) installed and running (`ollama run llama3`).
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed on your system.
- MongoDB instance (Atlas recommended for Vector Search).

### 2. Configuration
Create a `.env` file in the root directory:
```env
MONGO_URI=mongodb://localhost:27017/
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 3. Installation
```powershell
pip install -r requirements.txt
```

### 4. Running the App
Start the backend:
```powershell
python -m backend.main
```
Open `frontend/index.html` in your favorite browser.

## Tech Stack
- **Backend**: FastAPI, PyMuPDF, Pytesseract, LangChain, MongoDB.
- **Frontend**: HTML5, Tailwind CSS, Vanilla JS.
- **ML Models**: `sentence-transformers/all-MiniLM-L6-v2`.

## Data Schema (MongoDB)
Docs are stored in `remodel_catalog.chunks` with metadata including `tenant_id`, `page_number`, and `category`.


Modern kitchen with sleek grey cabinets and marble countertops."
"Minimalist bedroom design with wooden textures and warm lighting."
"Dark industrial kitchen layouts with hanging pendant lights."

"Show me kitchen designs featuring copper or gold hardware."
"Bedroom layouts with floor-to-ceiling windows and city views."
"Kitchen islands with built-in seating and waterfall edges."
"Plush upholstered headboards in neutral tones."


"What are the common design trends across the Kitchen Design Collection Book?"
"Compare the storage solutions shown in the kitchen catalog versus modern bedroom setups."
"Based on the catalogs, what is the best lighting setup for a small, compact kitchen?"




"Find the modern bedroom catalog and tell me what designs are featured on Page 10." (Testing the PDF linking)
"Are there any design references in the DesignBlenZ catalog that use blue accent walls?"