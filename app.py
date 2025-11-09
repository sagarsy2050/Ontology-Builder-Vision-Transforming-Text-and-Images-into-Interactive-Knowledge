# =============================
# Multi-Agent Vision-Enhanced Essay Generator (OLLAMA 2025)
# =============================
# pip install streamlit pandas ollama pillow tqdm
# ollama serve
# streamlit run app_vision.py

import streamlit as st
import json
import pandas as pd
import ollama
import subprocess
import time
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
import os

# =============================
# CONFIG
# =============================
TEXT_MODEL = "llama3.1:8b"
VISION_MODEL = "llava:13b"

# Multi-line prompt (no triple quotes)
ESSAY_PROMPT = (
    "You are **Agent: {name}**.\n"
    "Write a **200–300 word professional essay** about yourself.\n"
    "Follow these **directions**: {directions}\n"
    "Use these **relations** from the knowledge graph: {relations}\n"
    "{vision_context}\n"
    "Include sentiment (positive/negative/neutral) where relevant.\n"
    "Be clear, engaging, and technically accurate."
)

# =============================
# AUTO-PULL MODEL IF MISSING
# =============================
def ensure_model(model_name: str) -> bool:
    """Pull model if not found. Returns True if ready."""
    try:
        response = ollama.list()
        models = [m.get("name", "") for m in response.get("models", [])]
        if model_name in models:
            return True

        st.warning(f"Model `{model_name}` not found. Downloading now...")
        with st.spinner(f"Downloading `{model_name}` (~5–15 min)..."):
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                check=True
            )
        st.success(f"Model `{model_name}` downloaded!")
        return True
    except Exception as e:
        st.error(f"Failed to pull `{model_name}`: {e}")
        st.info("Run `ollama serve` in a terminal first.")
        return False

# =============================
# VISION CAPTION
# =============================
def get_image_caption(image_file) -> str:
    if not ensure_model(VISION_MODEL):
        return ""
    try:
        img = Image.open(image_file)
        resp = ollama.generate(
            model=VISION_MODEL,
            prompt="Describe this image in detail: focus on diagrams, text, concepts, and structure.",
            images=[img]
        )
        return resp.get("response", "").strip()
    except Exception as e:
        return f"[Vision error: {e}]"

# =============================
# LOAD JSON
# =============================
def load_ontology(file) -> Dict:
    try:
        return json.load(file)
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        return {}

# =============================
# EXTRACT AGENTS
# =============================
def extract_agents(ontology: Dict, source: str) -> List[Dict]:
    agents = []
    hierarchy = ontology.get("hierarchy", [])
    relations = ontology.get("relations", [])

    for item in hierarchy:
        concept = item.get("concept", "").strip()
        if not concept:
            continue

        related = item.get("related_concepts", [])
        directions = (
            f"Discuss connections with: {', '.join(related)}."
            if related else "Provide a self-contained overview."
        )

        rel_list = [
            f"{r['subject']} {r['relation']} {r['object']}"
            for r in relations
            if concept.lower() in str(r.get("subject", "")).lower()
            or concept.lower() in str(r.get("object", "")).lower()
        ]
        relations_str = "; ".join(rel_list) if rel_list else "No direct relations."

        agents.append({
            "name": concept,
            "directions": directions,
            "relations": relations_str,
            "source": source
        })
    return agents

# =============================
# GENERATE ESSAY
# =============================
def generate_essay(agent: Dict, vision_desc: str = "") -> str:
    if not ensure_model(TEXT_MODEL):
        return "[ERROR] Text model unavailable."

    vision_context = f"Based on image: {vision_desc}\n" if vision_desc else ""
    prompt = ESSAY_PROMPT.format(
        name=agent["name"],
        directions=agent["directions"],
        relations=agent["relations"],
        vision_context=vision_context
    )

    try:
        with st.spinner(f"Writing essay for **{agent['name']}**..."):
            resp = ollama.generate(model=TEXT_MODEL, prompt=prompt)
            return resp.get("response", "").strip()
    except Exception as e:
        return f"[ERROR] Generation failed: {e}"

# =============================
# STREAMLIT UI
# =============================
st.set_page_config(page_title="AI Agent Essay Generator", layout="wide")
st.title("Multi-Agent Vision-Enhanced Essay Generator")
st.markdown("**Upload ontology JSON + optional image → Generate 30+ AI agent essays**")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    uploaded_files = st.file_uploader(
        "Upload Ontology JSON(s)", type="json", accept_multiple_files=True
    )
    image_file = st.file_uploader("Upload Image (Optional)", type=["png", "jpg", "jpeg"])

    text_model = st.selectbox("Text Model", ["llama3.1:8b", "mistral:7b", "phi3:3.8b"])
    vision_model = st.selectbox("Vision Model", ["llava:13b", "qwen2-vl:7b"])

    if text_model != TEXT_MODEL:
        TEXT_MODEL = text_model
        st.rerun()
    if vision_model != VISION_MODEL:
        VISION_MODEL = vision_model
        st.rerun()

# --- Vision ---
vision_desc = ""
if image_file:
    vision_desc = get_image_caption(image_file)
    if vision_desc and not vision_desc.startswith("["):
        st.sidebar.success(f"Vision ON: {vision_desc[:100]}...")
    else:
        st.sidebar.warning("Vision failed or disabled.")

# --- Load Ontologies ---
all_agents = []
if uploaded_files:
    for idx, file in enumerate(uploaded_files):
        ontology = load_ontology(file)
        if not ontology:
            continue
        agents = extract_agents(ontology, f"File {idx+1}: {file.name}")
        all_agents.extend(agents)

    st.success(f"Loaded **{len(uploaded_files)}** file(s) → **{len(all_agents)} agents** extracted.")
else:
    st.info("Upload one or more `ontology_*.json` files to begin.")
    example = '{\n  "hierarchy": [{"concept": "quantum computing", "related_concepts": ["qubits"]}],\n  "relations": [{"subject": "quantum computing", "relation": "uses", "object": "qubits"}],\n  "summary": {"total_concepts": 5}\n}'
    st.markdown(f"### Expected JSON:\n```json\n{example}\n```")
    st.stop()

# --- Agent Table ---
if all_agents:
    df = pd.DataFrame(all_agents)
    st.subheader("Extracted Agents")
    st.dataframe(df[["name", "source"]], use_container_width=True)

    if st.button("Generate All Essays (One-by-One)", type="primary"):
        st.subheader("Generated Essays")
        progress = st.progress(0)
        status = st.empty()
        essays_md = "# Multi-Agent Vision-Enhanced Essays\n\n"

        for i, agent in enumerate(tqdm(all_agents, desc="Generating", leave=False)):
            status.text(f"Generating: {agent['name']} ({i+1}/{len(all_agents)})")
            essay = generate_essay(agent, vision_desc)

            with st.expander(f"**Agent: {agent['name']}**", expanded=True):
                st.markdown(essay)

            essays_md += f"## {agent['name']}\n\n{essay}\n\n---\n\n"
            progress.progress((i + 1) / len(all_agents))
            time.sleep(0.1)

        st.success("All essays generated!")
        st.download_button(
            "Download All Essays (Markdown)",
            essays_md,
            "multi_agent_essays.md",
            "text/markdown"
        )
        st.markdown(essays_md)

# --- Footer ---
st.markdown("---")
st.caption(f"Agents: **{len(all_agents)}** | Text: `{TEXT_MODEL}` | Vision: `{VISION_MODEL}` | Local Only")

