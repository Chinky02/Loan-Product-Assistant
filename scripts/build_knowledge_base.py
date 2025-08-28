import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm


def read_json_files(processed_dir: Path) -> List[Tuple[str, Dict]]:
    json_files = sorted(processed_dir.glob("*.json"))
    documents: List[Tuple[str, Dict]] = []
    for path in json_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            documents.append((str(path), data))
        except Exception as exc:
            print(f"[WARN] Failed to read {path}: {exc}")
    return documents


def flatten_json(obj: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, str]:
    items: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            if isinstance(v, (dict, list)):
                flat_child = flatten_json(v, new_key, sep)
                for ck, cv in flat_child.items():
                    items.append((ck, cv))
            else:
                if v is None:
                    continue
                items.append((new_key, str(v)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            if isinstance(v, (dict, list)):
                flat_child = flatten_json(v, new_key, sep)
                for ck, cv in flat_child.items():
                    items.append((ck, cv))
            else:
                if v is None:
                    continue
                items.append((new_key, str(v)))
    else:
        if obj is not None:
            items.append((parent_key or "value", str(obj)))
    return dict(items)


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text:
        return []
    text = normalize_whitespace(text)
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def collect_records(documents: List[Tuple[str, Dict]], min_len: int = 20) -> List[Dict]:
    records: List[Dict] = []
    for source_path, data in documents:
        flat = flatten_json(data)
        for key_path, value in flat.items():
            value = normalize_whitespace(value)
            if len(value) < min_len:
                continue
            records.append({
                "source": source_path,
                "key": key_path,
                "text": value,
            })
    return records


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb)
    if not embeddings:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    return np.vstack(embeddings).astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    if embeddings.size == 0:
        return faiss.IndexFlatIP(384)  # default dimension; will not be used
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_knowledge_base(output_dir: Path, index: faiss.Index, metadata: List[Dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS knowledge base from data/processed JSON files")
    parser.add_argument("--processed_dir", type=str, default=str(Path("data") / "processed"))
    parser.add_argument("--output_dir", type=str, default=str(Path("knowledge_base")))
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--chunk_overlap", type=int, default=120)
    parser.add_argument("--min_len", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)

    print(f"Reading JSON from: {processed_dir}")
    documents = read_json_files(processed_dir)
    if not documents:
        print("No JSON files found. Exiting.")
        return

    print("Collecting records...")
    base_records = collect_records(documents, min_len=args.min_len)

    print("Chunking records...")
    records: List[Dict] = []
    for rec in base_records:
        chunks = make_chunks(rec["text"], args.chunk_size, args.chunk_overlap)
        for idx, chunk in enumerate(chunks):
            records.append({
                "source": rec["source"],
                "key": rec["key"],
                "chunk_index": idx,
                "text": chunk,
            })

    texts = [r["text"] for r in records]

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    print("Computing embeddings...")
    embeddings = embed_texts(model, texts, batch_size=args.batch_size)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print(f"Saving knowledge base to: {output_dir}")
    save_knowledge_base(output_dir, index, [{k: v for k, v in r.items() if k != "text"} for r in records])

    print("Done.")


if __name__ == "__main__":
    main() 