import os
import shutil
import fitz
import re
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image
from io import BytesIO
from pymongo import MongoClient
from dotenv import load_dotenv
from backend.rag_tools import RAGTools

# ---------------- CONFIGURATION ----------------
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "remodel_catalog"
COLLECTION_NAME = "unified_nodes"

IMAGE_OUTPUT_DIR = "Data/processed/images"
OCR_OUTPUT_DIR = "Data/processed/ocr"
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)

# OCR Setup
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

rag = RAGTools()
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# ---------------- HELPERS ----------------

def is_valid_image(pil_img):
    w, h = pil_img.size
    if w < 150 or h < 150 or (w * h) < 30000:
        return False
    try:
        img_arr = np.array(pil_img.convert("L").resize((50,50)))
        if np.var(img_arr) < 100: return False
    except: return False
    return True

def clean_filename(name):
    return re.sub(r'[^a-zA-Z0-9]', '_', name)

def parse_txt_catalog(file_path):
    if not file_path or not os.path.exists(file_path):
        return []
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by "Page X |"
    entries = []
    # Using regex to find starts of entries
    # Format: Page 1 | Kitchen Layout 1-1 Style: ...
    pattern = r"Page (\d+) \| (.*?)(?=Page \d+ \||\Z)"
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        page_no = int(match.group(1))
        block = match.group(2).strip()
        
        # Parse fields within block
        # Example: Kitchen Layout 1-1 Style: Urban Material: Plywood + Laminate ...
        entry = {"page": page_no}
        
        # Extract Product Name (it's everything before "Style:")
        name_match = re.match(r"(.*?) Style:", block)
        if name_match:
            entry["product"] = name_match.group(1).strip()
        else:
            continue
            
        # Regex for common fields
        fields = [
            ("style", r"Style: (.*?) Material:"),
            ("material", r"Material: (.*?) Color:"),
            ("color", r"Color: (.*?) (?:Size|Layout Size):"),
            ("size", r"(?:Size|Layout Size): (.*?) Warranty:"),
            ("warranty", r"Warranty: (.*?) Delivery:"),
            ("delivery", r"Delivery: (.*?) Installation:"),
            ("installation", r"Installation: (.*?) Description:"),
            ("description", r"Description: (.*?) Price:"),
            ("price", r"Price: (.*?)$")
        ]
        
        for key, field_pattern in fields:
            f_match = re.search(field_pattern, block, re.DOTALL)
            if f_match:
                entry[key] = f_match.group(1).strip().replace("\n", " ")
        
        entries.append(entry)
        
    return entries

# ---------------- PROCESSING ----------------

def extract_pdf_images(pdf_path, pdf_label):
    doc = fitz.open(pdf_path)
    page_images = {} # page_no -> [ {path, ocr, embedding} ]
    
    clean_label = clean_filename(pdf_label)
    
    for page_idx in range(len(doc)):
        page_no = page_idx + 1
        page = doc[page_idx]
        image_list = page.get_images(full=True)
        valid_imgs = []
        
        for img_idx, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                pil_img = Image.open(BytesIO(base_image["image"]))
                
                if not is_valid_image(pil_img): continue
                
                # Force RGB mode (to handle CMYK or other non-RGB source formats)
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                
                # Force PNG extension
                ext = "png"
                img_name = f"{clean_label}_p{page_no}_i{img_idx}.{ext}"
                img_path = os.path.join(IMAGE_OUTPUT_DIR, img_name)
                
                # Save as PNG
                pil_img.save(img_path, format="PNG")
                
                # Perform OCR (Fallback to page text if Tesseract unavailable)
                ocr_text = ""
                try:
                    ocr_text = pytesseract.image_to_string(pil_img).strip()
                except Exception as tess_err:
                    # If Tesseract fails, we'll use the page's text as a context proxy later, 
                    # but for now we leave it empty and logging the issue.
                    pass
                
                # Save OCR to txt file (Always save at least an empty one or a page marker if Tesseract failed)
                ocr_filename = f"{img_name}.txt"
                ocr_path = os.path.join(OCR_OUTPUT_DIR, ocr_filename)
                with open(ocr_path, "w", encoding="utf-8") as f:
                    f.write(ocr_text)
                
                # Get CLIP embedding
                clip_emb = None
                if img_path and os.path.exists(img_path):
                    try:
                        clip_emb = rag.get_clip_image_embedding(img_path)
                    except Exception as emb_err:
                        print(f"   |_ [WARN] Embedding failed for {img_name}: {emb_err}")
                
                valid_imgs.append({
                    "path": img_path,
                    "ocr_text": ocr_text,
                    "clip_embedding": clip_emb
                })
            except Exception as e:
                print(f"Error extracting image {img_idx} on page {page_no}: {e}")
                continue
        
        page_images[page_no] = valid_imgs
        
    doc.close()
    return page_images

def process_job(job):
    category = job["category"]
    print(f"\n>>> PROCESSING: {category.upper()}")
    
    # 1. Extract Images from PDF
    pdf_images = extract_pdf_images(job["pdf"], category)
    
    # 2. Parse TXT Catalog
    txt_entries = parse_txt_catalog(job["txt"])
    print(f"Found {len(txt_entries)} entries in TXT catalog.")
    
    # 3. Combine and Store
    total_pushed = 0
    for entry in txt_entries:
        page_no = entry["page"]
        product_name = entry.get("product", "Unknown")
        
        # Link Images from the same page
        images_on_page = pdf_images.get(page_no, [])
        image_paths = [img["path"].replace("\\", "/") for img in images_on_page]
        
        # Construct ID
        clean_prod = clean_filename(product_name).lower()
        node_id = f"{category}_{page_no}_{clean_prod}"
        
        # Combined text for embedding
        fields_to_combine = [
            f"Product: {product_name}",
            f"Category: {category}",
            f"Style: {entry.get('style', '')}",
            f"Material: {entry.get('material', '')}",
            f"Color: {entry.get('color', '')}",
            f"Size: {entry.get('size', '')}",
            f"Description: {entry.get('description', '')}",
            f"Price: {entry.get('price', '')}"
        ]
        
        # Include OCR from related images in combined text for better search
        for img in images_on_page:
            if img["ocr_text"]:
                fields_to_combine.append(f"Image Content: {img['ocr_text']}")
                
        combined_text = " | ".join(fields_to_combine)
        embedding = rag.get_embeddings(combined_text)
        
        # Final Document
        doc = {
            "id": node_id,
            "category": category,
            "page": page_no,
            "product": product_name,
            "style": entry.get("style", ""),
            "material": entry.get("material", ""),
            "color": entry.get("color", ""),
            "size": entry.get("size", ""),
            "price": entry.get("price", ""),
            "warranty": entry.get("warranty", ""),
            "delivery": entry.get("delivery", ""),
            "installation": entry.get("installation", ""),
            "description": entry.get("description", ""),
            "image_paths": image_paths, # List of strings as requested
            "related_images": images_on_page, # Storing full objects inclusive of OCR/Embeddings internally
            "combined_text": combined_text,
            "embedding": embedding
        }
        
        collection.replace_one({"id": node_id}, doc, upsert=True)
        total_pushed += 1

    print(f"Pushed {total_pushed} unified nodes for {category}.")

def clear_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def ingest_all():
    print(f"Clearing collection: {COLLECTION_NAME}")
    collection.delete_many({})
    
    # Also clear processed directories to ensure "clear images and text"
    import shutil
    clear_directory(IMAGE_OUTPUT_DIR)
    clear_directory(OCR_OUTPUT_DIR)

    jobs = [
        {
            "category": "automotive",
            "pdf": "Data/Automotive_Catalogue_4_April_2025.pdf",
            "txt": None
        }
    ]

    for job in jobs:
        # If no TXT provided, we'll create entries based on PDF text extraction + Image discovery
        if not job.get("txt"):
            print(f"No manual catalog TXT for {job['category']}. Using PDF-text and Image discovery.")
            
            doc_pdf = fitz.open(job["pdf"])
            pdf_images = extract_pdf_images(job["pdf"], job["category"])
            total_pushed = 0
            
            for page_idx in range(len(doc_pdf)):
                page_no = page_idx + 1
                page_text = doc_pdf[page_idx].get_text().strip()
                images = pdf_images.get(page_no, [])
                
                if not page_text and not images: continue
                
                node_id = f"{job['category']}_p{page_no}"
                image_paths = []
                for img in images:
                    if img.get("path"):
                        image_paths.append(img["path"].replace("\\", "/"))
                
                # Refine category detection for this specific page
                p_lower = page_text.lower()
                category = job["category"]
                if "impact wrench" in p_lower or "pneumatic" in p_lower:
                    category = "Pneumatic Tools"
                elif "soldering" in p_lower or "welding" in p_lower:
                    category = "Soldering & Welding"
                elif "electrical" in p_lower or "wiring" in p_lower or "heat-shrink" in p_lower:
                    category = "Wiring & Electrical"
                elif "heavy duty" in p_lower or "truck" in p_lower:
                    category = "Heavy Duty Service"

                # Combine all available text for this page
                combined_text = f"Category: {category} | Page: {page_no} | Text: {page_text}"
                for img in images:
                    # If image OCR is empty (Tesseract failed), we inject page text segment as proxy
                    if not img.get("ocr_text") and page_text:
                        img["ocr_text"] = f"[Context: {page_text[:100]}...]"
                        # Update the txt file with this context proxy
                        ocr_path = os.path.join(OCR_OUTPUT_DIR, f"{os.path.basename(img['path'])}.txt")
                        with open(ocr_path, "w", encoding="utf-8") as f:
                            f.write(img["ocr_text"])

                    if img.get("ocr_text"):
                        combined_text += f" | Image Content: {img['ocr_text']}"
                
                # Broadened Art. No. Search (e.g., 083014 750 or 0714 661 113)
                art_no_match = re.search(r"Art\. No\. (\d{3,6}[\s\-]?\d{3}[\s\-]?\d{2,4})", page_text)
                product_label = f"Automotive Product - Page {page_no}"
                if art_no_match:
                    art_id = art_no_match.group(1).strip()
                    product_label = f"WURTH Product - Art. No. {art_id}"
                    # Explicitly inject into combined text for high-weight indexing
                    combined_text = f"ArtNo: {art_id} | {combined_text}"

                doc = {
                    "id": node_id,
                    "category": category,
                    "page": page_no,
                    "product": product_label,
                    "description": page_text[:2000],
                    "image_paths": image_paths,
                    "related_images": images,
                    "combined_text": combined_text,
                    "embedding": rag.get_embeddings(combined_text[:1000])
                }
                collection.replace_one({"id": node_id}, doc, upsert=True)
                total_pushed += 1
            
            doc_pdf.close()
            print(f"Pushed {total_pushed} nodes for {job['category']}.")
        else:
            process_job(job)

if __name__ == "__main__":
    try:
        ingest_all()
        print("\nINGESTION COMPLETE!")
    except Exception as e:
        print(f"Ingestion failed: {e}")
