import streamlit as st
import os
import requests
from dotenv import load_dotenv
from PIL import Image

# ---------------- CONFIGURATION ----------------
load_dotenv()

st.set_page_config(
    page_title="Wurthi Automotive Solutions",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium "White Glassmorphism" Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&family=Playfair+Display:ital,wght@0,700;1,700&display=swap');

    /* Global */
    .stApp {
        background: radial-gradient(circle at top left, #f8faff 0%, #eef2ff 50%, #f1f5f9 100%);
        color: #1e293b;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    /* Typography */
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 5rem !important;
        background: linear-gradient(135deg, #020617 0%, #1e293b 50%, #64748b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px !important;
        font-weight: 700;
        letter-spacing: -2px;
    }

    .hero-subtitle {
        font-family: 'Outfit', sans-serif;
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        letter-spacing: 6px;
        text-transform: uppercase;
        margin-bottom: 3.5rem;
        font-weight: 400;
    }

    /* Glass Cards */
    .glass-panel {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(25px) saturate(180%);
        -webkit-backdrop-filter: blur(25px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 32px;
        padding: 45px;
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.05),
                    inset 0 0 0 1px rgba(255, 255, 255, 0.5);
    }

    /* Input Field */
    .stTextInput input {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 2px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 16px !important;
        color: #0f172a !important;
        padding: 1.2rem 1.8rem !important;
        font-family: 'Outfit', sans-serif;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
    }

    .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.15) !important;
        background: #ffffff !important;
    }

    /* Buttons */
    .stButton>button {
        background: #0f172a !important;
        color: #ffffff !important;
        border-radius: 14px !important;
        padding: 0.8rem 2.5rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        border: none !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 10px 20px -5px rgba(15, 23, 42, 0.3);
    }

    .stButton>button:hover {
        background: #1e293b !important;
        transform: translateY(-3px);
        box-shadow: 0 15px 30px -8px rgba(15, 23, 42, 0.4);
    }

    /* Badges */
    .status-badge {
        background: rgba(15, 23, 42, 0.05);
        color: #0f172a;
        border: 1px solid rgba(15, 23, 42, 0.1);
        padding: 6px 16px;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 15px;
        border: 1px solid #f1f5f9;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
    }
</style>
""", unsafe_allow_html=True)

# ---------------- API CLIENT ----------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def ask_api(question, rag_mode):
    try:
        payload = {"question": question, "rag_mode": rag_mode}
        response = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=300)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

# ---------------- HERO SECTION ----------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Wurthi Automotive Solutions</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Industrial Vehicle Discovery & Component Intelligence</p>', unsafe_allow_html=True)

# Search Logic
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    # RAG Mode Selector
    st.markdown("### 🧠 Intelligence Mode")
    rag_mode = st.selectbox(
        "Select RAG Strategy",
        options=["auto", "standard", "corrective", "speculative", "fusion"],
        format_func=lambda x: {
            "auto": "🤖 Auto - Smart Router",
            "standard": "⚡ Standard - Fast & Efficient",
            "corrective": "🎯 Corrective - Quality-Aware",
            "speculative": "🔬 Speculative - Deep Analysis",
            "fusion": "🌊 Fusion - High Recall"
        }[x],
        index=0,
        label_visibility="collapsed"
    )
    
    mode_descriptions = {
        "auto": "Best for complex technical queries - chooses the optimal path (~3s)",
        "standard": "Lightning-fast retrieval for simple part lookups (~1s)",
        "corrective": "Verifies results against catalog specs for high precision (~4s)",
        "speculative": "Generates detailed technical drafts and enriches with entity extraction (~4s)",
        "fusion": "Multi-query variations for finding components across different catalog sections (~5s)"
    }
    st.caption(mode_descriptions[rag_mode])
    st.markdown("<br>", unsafe_allow_html=True)
    
    query = st.text_input("Component Search", placeholder="Search for technical components (e.g., '1-inch pneumatic impact wrench 2440 Nm')", label_visibility="collapsed")
    
    st.markdown('<div style="display:flex; gap:10px; margin-top:15px; flex-wrap:wrap;">', unsafe_allow_html=True)
    suggestions = ["High torque wrenches", "Heat-shrink sets", "Soldering devices", "Impact sockets"]
    for s in suggestions:
        if st.button(s, key=f"sug_{s}"):
            query = s
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col2:
        search_triggered = st.button("DISCOVER")

st.markdown("---")

# ---------------- RESULTS ----------------
if search_triggered or query:
    if query:
        with st.status("🧠 Processing Automotive Intelligence...", expanded=False) as status:
            try:
                result = ask_api(query, rag_mode=rag_mode)
                
                if "error" in result:
                    st.error(result["error"])
                    status.update(label="Analysis failed", state="error")
                else:
                    status.update(label=f"Analysis complete ({result.get('generation_time', 'N/A')})", state="complete")
                
                # Layout
                left_col, right_col = st.columns([1, 1], gap="large")
                
                with left_col:
                    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                    st.markdown("### 🔍 Technical Analysis")
                    st.write(result["answer"])
                    
                    # Mode-specific metadata badges
                    st.markdown('<br>', unsafe_allow_html=True)
                    badges = ['<span class="status-badge">WURTH-SPECIFIED</span>', '<span class="status-badge">CATALOG-MATCHED</span>']
                    
                    if result.get("mode") == "corrective":
                        score = float(result.get("relevance_score", 0))
                        color = "#10b981" if score > 0.6 else "#f97316"
                        badges.append(f'<span class="status-badge" style="background: rgba(245, 158, 11, 0.1); color: {color}; border-color: {color};">RELEVANCE: {score:.2f}</span>')
                    elif result.get("mode") == "speculative":
                        entities = result.get("entities", [])
                        if entities:
                            entity_str = ", ".join(entities[:3])
                            badges.append(f'<span class="status-badge" style="background: rgba(139, 92, 246, 0.1); color: #8b5cf6; border-color: #8b5cf6;">ENTITIES: {entity_str}</span>')
                    
                    st.markdown(' '.join(badges), unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with right_col:
                    st.markdown("### 🖼️ Catalog Discoveries")
                    images = result.get("images", [])
                    if images:
                        img_grid = st.columns(2)
                        for i, img_data in enumerate(images[:6]):
                            with img_grid[i % 2]:
                                # Try to find a valid display path
                                path_from_db = img_data.get("image_path")
                                display_path = None
                                
                                if path_from_db:
                                    if os.path.isfile(path_from_db):
                                        display_path = path_from_db
                                    else:
                                        # Reconstruction fallback for relative paths
                                        fname = os.path.basename(path_from_db.replace("\\", "/"))
                                        potential = os.path.join("Data", "processed", "images", fname)
                                        if os.path.isfile(potential):
                                            display_path = potential

                                # Render Image if found
                                if display_path:
                                    try:
                                        img_obj = Image.open(display_path)
                                        st.image(img_obj, use_container_width=True)
                                    except Exception as img_err:
                                        st.warning(f"Error rendering image: {img_err}")
                                else:
                                    st.caption(f"⚠️ Media Not Found: {os.path.basename(path_from_db) if path_from_db else 'N/A'}")

                                # Display Metadata (simplified)
                                pg_num = img_data.get("page", "N/A")
                                st.markdown(f'<p style="font-size:0.9rem; color:#f8fafc; text-align:center; font-weight:bold; margin-bottom:5px;">Page {pg_num}</p>', unsafe_allow_html=True)
                                
                                pdf_url = img_data.get("pdf_url")
                                if pdf_url:
                                    st.markdown(f'''
                                        <div style="text-align:center; margin-top:2px;">
                                            <a href="{pdf_url}" target="_blank" 
                                               style="background:rgba(59, 130, 246, 0.1); color:#60a5fa; border:1px solid rgba(59, 130, 246, 0.3); 
                                               padding:5px 20px; border-radius:50px; text-decoration:none; font-size:0.75rem; font-weight:bold; 
                                               display:inline-block; transition:all 0.3s ease;">
                                               OPEN PAGE {pg_num} 🔩
                                            </a>
                                        </div>''', unsafe_allow_html=True)
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                    else:
                        st.info("🎨 Note: We couldn't find exact catalog images for this specific query, but our AI analysis above describes the best design approach based on our architectural knowledge.")
                
            except Exception as e:
                st.error(f"Neural linkage error: {e}")
    else:
        st.warning("Please provide a vision to begin discovery.")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#334155; letter-spacing:2px; font-size:0.7rem;'>INFINIA AI // POWERED BY CLIP MODEL // 2026 AUTOMOTIVE CATALOG V1</div>", unsafe_allow_html=True)
