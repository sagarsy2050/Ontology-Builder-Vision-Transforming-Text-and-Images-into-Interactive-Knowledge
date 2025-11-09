# =============================
# LOCAL ONTOLOGY BUILDER + SENTIMENT + VISION (OLLAMA 2025)
# =============================
# pip install spacy keybert sentence-transformers networkx ollama pillow tqdm
# python -m spacy download en_core_web_sm
# ollama pull llama3.1:8b  # Text model
# ollama pull llava:13b     # Vision model (2025 recommended)

import json
import spacy
import subprocess
import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
import networkx as nx
import ollama
from PIL import Image
from tqdm import tqdm

# =============================
# LOGGING SETUP
# =============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# =============================
# CONFIGURATION (2025 UPDATE)
# =============================
TEXT_MODEL = "llama3.1:8b"  # Efficient text model
VISION_MODEL = "llava:13b"  # Vision model (LLaVA 1.6, high-res, OCR-strong)
SIMILARITY_THRESHOLD = 0.6
TOP_N_CONCEPTS = 30
SENTIMENT_PROMPT = (
    "Analyze the sentiment of this statement. "
    "Return only: positive, negative, or neutral. "
    "Do not explain.\n\nStatement: {stmt}"
)

# =============================
# AUTO-INSTALL SPACY MODEL
# =============================
def ensure_spacy_model(model_name: str = "en_core_web_sm"):
    try:
        spacy.load(model_name)
        log.info(f"spaCy model '{model_name}' ready.")
    except OSError:
        log.warning(f"spaCy model '{model_name}' missing. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            log.info(f"Installed '{model_name}'.")
        except Exception as e:
            log.error(f"Failed to install spaCy model: {e}")
            raise

ensure_spacy_model()

# =============================
# SAFE OLLAMA CALL (TEXT + VISION)
# =============================
def safe_ollama_generate(prompt: str, model: str, images=None) -> str:
    try:
        resp = ollama.generate(
            model=model,
            prompt=prompt,
            images=images or [],
            options={"temperature": 0.0}
        )
        return resp.get("response", "").strip()
    except Exception as e:
        log.warning(f"Ollama call failed ({model}): {e}")
        return ""

# =============================
# STEP 1: Load Text (Robust)
# =============================
def load_text(file_path: Optional[str] = None, raw_text: Optional[str] = None) -> str:
    if file_path:
        path = Path(file_path)
        if not path.exists():
            log.error(f"File not found: {path}")
            raise FileNotFoundError(path)
        return path.read_text(encoding="utf-8").strip()
    elif raw_text:
        return raw_text.strip()
    else:
        raise ValueError("Must provide 'file_path' or 'raw_text'")

# =============================
# STEP 2: Extract Concepts
# =============================
def extract_concepts(text: str) -> List[str]:
    log.info("Extracting concepts...")
    try:
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=TOP_N_CONCEPTS
        )
        concepts = [kw[0].strip() for kw in keywords if kw[0].strip()]
        concepts = list(dict.fromkeys(concepts))
        log.info(f"Extracted {len(concepts)} concepts.")
        return concepts
    except Exception as e:
        log.error(f"Concept extraction failed: {e}")
        return []

# =============================
# STEP 3: Extract Relations
# =============================
def extract_relations(text: str) -> List[Dict[str, str]]:
    log.info("Extracting relations...")
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        triples = []

        for sent in doc.sents:
            root = sent.root
            subject = object_ = None

            for child in root.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = " ".join([t.text for t in child.subtree]).strip()
                elif child.dep_ in ("dobj", "iobj", "pobj", "attr", "dative"):
                    object_ = " ".join([t.text for t in child.subtree]).strip()

            if subject and object_:
                triples.append({
                    "subject": subject,
                    "relation": root.lemma_,
                    "object": object_,
                    "sentence": sent.text.strip()
                })

        log.info(f"Extracted {len(triples)} raw triples.")
        return triples
    except Exception as e:
        log.error(f"Relation extraction failed: {e}")
        return []

# =============================
# STEP 4: Sentiment + Confidence Scoring
# =============================
def analyze_sentiment(statement: str) -> Tuple[str, float]:
    prompt = SENTIMENT_PROMPT.format(stmt=statement)
    result = safe_ollama_generate(prompt, TEXT_MODEL)
    result = result.lower()

    if "positive" in result:
        return "positive", 1.0
    elif "negative" in result:
        return "negative", -1.0
    else:
        return "neutral", 0.0

def enrich_triples(triples: List[Dict]) -> List[Dict]:
    log.info("Enriching triples with sentiment...")
    enriched = []
    for t in tqdm(triples, desc="Sentiment", leave=False):
        stmt = f"{t['subject']} {t['relation']} {t['object']}"
        sentiment, score = analyze_sentiment(stmt)
        t.update({
            "sentiment": sentiment,
            "polarity_score": score,
            "confidence": abs(score)
        })
        enriched.append(t)
    return enriched

# =============================
# STEP 5: Build Graph
# =============================
def build_concept_graph(concepts: List[str]) -> nx.Graph:
    log.info("Building concept graph...")
    if not concepts:
        return nx.Graph()

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(concepts, convert_to_tensor=True)
        G = nx.Graph()

        for c in concepts:
            G.add_node(c)

        edges = 0
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                if sim > SIMILARITY_THRESHOLD:
                    G.add_edge(concepts[i], concepts[j], weight=round(sim, 4))
                    edges += 1

        log.info(f"Graph built with {edges} edges.")
        return G
    except Exception as e:
        log.error(f"Graph build failed: {e}")
        return nx.Graph()

# =============================
# STEP 6: Image Caption (Ollama Vision 2025)
# =============================
def caption_image(image_path: str) -> str:
    log.info(f"Captioning image with {VISION_MODEL}: {image_path}")
    try:
        img = Image.open(image_path)
        resp = safe_ollama_generate(
            "Describe this image in detail, focusing on key elements, diagrams, and text (OCR if present). Keep concise but informative.",
            VISION_MODEL,
            images=[img]
        )
        caption = resp or "[No caption generated]"
        log.info(f"Vision caption: {caption}")
        return caption
    except Exception as e:
        err = f"[Vision error: {e}]"
        log.warning(err)
        return err

# =============================
# STEP 7: Export
# =============================
def export_ontology(
    concepts: List[str],
    triples: List[Dict],
    graph: nx.Graph
) -> Dict[str, Any]:
    hierarchy = [
        {"concept": n, "related_concepts": list(graph.neighbors(n))}
        for n in graph.nodes
    ]

    return {
        "concepts": concepts,
        "relations": triples,
        "hierarchy": hierarchy,
        "summary": {
            "total_concepts": len(concepts),
            "total_relations": len(triples),
            "positive_triples": len([t for t in triples if t.get("sentiment") == "positive"]),
            "negative_triples": len([t for t in triples if t.get("sentiment") == "negative"]),
            "neutral_triples": len([t for t in triples if t.get("sentiment") == "neutral"]),
            "graph_edges": graph.number_of_edges()
        }
    }

# =============================
# MAIN PIPELINE
# =============================
def build_ontology(
    raw_text: Optional[str] = None,
    file_path: Optional[str] = None,
    image_path: Optional[str] = None
) -> Dict[str, Any]:
    log.info("Starting ontology build...")

    try:
        text = load_text(file_path=file_path, raw_text=raw_text)
    except Exception as e:
        log.critical(f"Input failed: {e}")
        return {"error": str(e)}

    # Vision Integration: Caption image if provided
    if image_path:
        caption = caption_image(image_path)
        text += f"\n\n[Vision Description]: {caption}"

    concepts = extract_concepts(text)
    raw_triples = extract_relations(text)
    enriched_triples = enrich_triples(raw_triples)
    graph = build_concept_graph(concepts)
    ontology = export_ontology(concepts, enriched_triples, graph)

    log.info("Ontology build completed successfully.")
    return ontology

# =============================
# EXAMPLE USAGE
# =============================
if __name__ == "__main__":
    sample_text = """
    Quantum computing is a revolutionary technology that uses qubits instead of bits.
    Qubits can be in superposition, which enables massive parallelism.
    However, quantum systems are extremely sensitive to noise and decoherence.
    Variational Quantum Algorithms help mitigate errors but require many iterations.
    Quantum advantage has not been fully proven in real-world applications.
    """

    ontology = build_ontology(raw_text=sample_text)  # Add image_path="quantum_diagram.png"

    out_file = "ontology_vision.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ontology, f, indent=4, ensure_ascii=False)

    print(f"\nOntology saved: {out_file}")
    print(json.dumps(ontology["summary"], indent=2))