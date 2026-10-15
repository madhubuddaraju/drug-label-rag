"""Paths and settings shared by the pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
LABELS = DATA / "labels.jsonl.gz"
LABELS_META = DATA / "labels_meta.json"
CHUNKS = DATA / "chunks.jsonl.gz"
EMBEDDINGS = DATA / "embeddings.npy"          # float16, one row per chunk, L2-normalised
EVALS = ROOT / "evals"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

SEED = 42
N_DRUGS = 600                  # most widely marketed prescription generics (by number of distinct labels)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_MODEL_WEB = "Xenova/bge-small-en-v1.5"   # ONNX port of the same weights, used by the browser demo
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

CHUNK_WORDS = 160
CHUNK_OVERLAP = 30

# openFDA field -> (short key, display title). Order is the order sections appear in a label.
SECTIONS = {
    "boxed_warning": ("boxed_warning", "Boxed warning"),
    "indications_and_usage": ("indications", "Indications and usage"),
    "dosage_and_administration": ("dosage", "Dosage and administration"),
    "contraindications": ("contraindications", "Contraindications"),
    "warnings_and_cautions": ("warnings", "Warnings and precautions"),
    "warnings": ("warnings", "Warnings and precautions"),
    "adverse_reactions": ("adverse_reactions", "Adverse reactions"),
    "drug_interactions": ("interactions", "Drug interactions"),
    "pregnancy": ("pregnancy", "Pregnancy"),
    "nursing_mothers": ("lactation", "Lactation"),
    "lactation": ("lactation", "Lactation"),
    "pediatric_use": ("pediatric", "Pediatric use"),
    "geriatric_use": ("geriatric", "Geriatric use"),
    "overdosage": ("overdosage", "Overdosage"),
}
SECTION_TITLES = {key: title for key, title in SECTIONS.values()}
