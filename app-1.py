import streamlit as st
import json
import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
import networkx as nx
import ollama
from PIL import Image
from pathlib import Path

# =============================
# CONFIG
# =============================
TEXT_MODEL = "llama3.1:8b"
VISION_MODEL = "llava:13b"
SIMILARITY_THRESHOLD = 0.6
TOP_N_CONCEPTS = 30

# Load models once
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    kw_model = KeyBERT()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp, kw_model, embedder

nlp, kw_model, embedder = load_models()

# =============================
# HELPERS
# =============================
def safe_ollama(prompt, model, images=None):
    try:
        resp = ollama.generate(model=model, prompt=prompt, images=images or [], options={"temperature": 0.0})
        return resp["response"].strip()
    except:
        return ""

def extract_concepts(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=TOP_N_CONCEPTS)
    return list(dict.fromkeys([kw[0].strip() for kw in keywords if kw[0].strip()]))

def extract_relations(text):
    doc = nlp(text)
    triples = []
    for sent in doc.sents:
        root = sent.root
        subject = object_ = None
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subject = " ".join([t.text for t in child.subtree])
            elif child.dep_ in ("dobj", "pobj", "attr", "dative"):
                object_ = " ".join([t.text for t in child.subtree])
        if subject and object_:
            triples.append({"subject": subject, "relation": root.lemma_, "object": object_, "sentence": sent.text})
    return triples

def analyze_sentiment(stmt):
    prompt = f"Return only: positive, negative, or neutral.\nStatement: {stmt}"
    result = safe_ollama(prompt, TEXT_MODEL).lower()
    if "positive" in result: return "positive", 1.0
    if "negative" in result: return "negative", -1.0
    return "neutral", 0.0

def enrich_triples(triples):
    for t in triples:
        stmt = f"{t['subject']} {t['relation']} {t['object']}"
        sent, score = analyze_sentiment(stmt)
        t.update({"sentiment": sent, "polarity_score": score, "confidence": abs(score)})
    return triples

def build_graph(concepts):
    if not concepts: return nx.Graph()
    embeddings = embedder.encode(concepts, convert_to_tensor=True)
    G = nx.Graph()
    G.add_nodes_from(concepts)
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim > SIMILARITY_THRESHOLD:
                G.add_edge(concepts[i], concepts[j], weight=round(sim, 4))
    return G

def caption_image(img):
    resp = safe_ollama("Describe this image: key elements, diagrams, text. Be concise.", VISION_MODEL, images=[img])
    return resp or "[No caption]"

# =============================
# STREAMLIT UI
# =============================
st.set_page_config(page_title="Local Ontology Builder", layout="wide")
st.title("Local Ontology Builder + Vision (LLaMA 3.1 + LLaVA)")

col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("Enter text to analyze:", height=200, placeholder="Paste your document or description here...")
    
with col2:
    image_file = st.file_uploader("Upload image (optional)", type=["png", "jpg", "jpeg"])

if st.button("Build Ontology", type="primary"):
    if not text_input.strip():
        st.error("Please enter some text.")
    else:
        with st.spinner("Building ontology..."):
            text = text_input.strip()
            if image_file:
                img = Image.open(image_file)
                st.image(img, caption="Uploaded Image", width=200)
                caption = caption_image(img)
                st.info(f"Vision: {caption}")
                text += f"\n\n[Image Description]: {caption}"

            concepts = extract_concepts(text)
            triples = extract_relations(text)
            triples = enrich_triples(triples)
            graph = build_graph(concepts)
            
            hierarchy = [{"concept": n, "related": list(graph.neighbors(n))} for n in graph.nodes]
            summary = {
                "concepts": len(concepts),
                "relations": len(triples),
                "positive": len([t for t in triples if t["sentiment"] == "positive"]),
                "negative": len([t for t in triples if t["sentiment"] == "negative"]),
                "neutral": len([t for t in triples if t["sentiment"] == "neutral"]),
                "graph_edges": graph.number_of_edges()
            }

            ontology = {
                "concepts": concepts,
                "relations": triples,
                "hierarchy": hierarchy,
                "summary": summary
            }

            st.success("Ontology built!")
            st.json(ontology, expanded=False)

            # Download
            st.download_button(
                "Download JSON",
                data=json.dumps(ontology, indent=2),
                file_name="ontology.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.caption("Powered by Ollama (LLaMA 3.1 + LLaVA) | Local & Private | No cloud.")