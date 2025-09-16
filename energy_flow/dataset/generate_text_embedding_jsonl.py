#!/usr/bin/env python3
"""
Generate JSONL dataset with DistilBERT embeddings (Windows/NVIDIA friendly)

Each output line is a JSON object with fields:
  {
    "text": "...",
    "input_ids": [int, ...],
    "attention_mask": [0/1, ...],
    "target": [float, ... 768 floats ...]
  }

Target dimension equals the reference model hidden size (default: distilbert-base-uncased, 768).
Pooling: masked mean over last_hidden_state to produce a single vector per text.

Requirements (install on Windows/NVIDIA):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install transformers tqdm pandas

Usage examples:
  # Plain text file (one text per line)
  python generate_text_embedding_jsonl.py --input texts.txt --output out.jsonl

  # JSONL with a "text" field
  python generate_text_embedding_jsonl.py --input data.jsonl --output out.jsonl --text-column text

  # CSV with a "text" column
  python generate_text_embedding_jsonl.py --input data.csv --output out.jsonl --text-column text

  # Directory of .txt files
  python generate_text_embedding_jsonl.py --input ./texts_dir --output out.jsonl

Notes:
- The script automatically uses CUDA if available; otherwise falls back to CPU.
- Set --max-length to control tokenization length (default 128).
- Set --batch-size for throughput (32 by default).
- Output is appended if --append is provided.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

try:
    import pandas as pd  # optional, used for CSV
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False


def iter_texts_from_path(path: Path, text_column: str = "text") -> Iterable[str]:
    """Yield texts from a file or directory.
    Supports: .txt (one line per sample), .jsonl (expects {text_column: str}), .csv (text_column), or directory of .txt files.
    """
    if path.is_dir():
        for p in sorted(path.glob("*.txt")):
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            if txt:
                yield txt
        return

    suffix = path.suffix.lower()
    if suffix == ".txt":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                t = line.strip()
                if t:
                    yield t
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    t = str(obj.get(text_column, "")).strip()
                    if t:
                        yield t
                except Exception:
                    continue
    elif suffix == ".csv":
        if not HAS_PANDAS:
            raise RuntimeError("pandas is required to read CSV. Install with: pip install pandas")
        df = pd.read_csv(path)
        if text_column not in df.columns:
            raise RuntimeError(f"CSV missing column '{text_column}'")
        for t in df[text_column].astype(str).tolist():
            t = t.strip()
            if t:
                yield t
    else:
        raise RuntimeError(f"Unsupported input type: {path}")


def batched(iterable: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def masked_mean(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute masked mean over sequence length.
    last_hidden_state: [B, L, H]
    attention_mask:   [B, L]
    returns:          [B, H]
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B,L,1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B,H]
    denom = mask.sum(dim=1).clamp(min=1e-9)                         # [B,1]
    return summed / denom


def process_inputs(
    inputs: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int,
) -> Tuple[List[Dict[str, List[int]]], List[List[float]]]:
    enc = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        pooled = masked_mean(last_hidden, attention_mask)  # [B, H]

    pooled = pooled.detach().to("cpu")
    input_ids_cpu = input_ids.detach().to("cpu")
    attention_cpu = attention_mask.detach().to("cpu")

    meta: List[Dict[str, List[int]]] = []
    for i in range(pooled.size(0)):
        meta.append({
            "input_ids": input_ids_cpu[i].tolist(),
            "attention_mask": attention_cpu[i].tolist(),
        })

    vectors: List[List[float]] = pooled.tolist()
    return meta, vectors


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Generate JSONL with DistilBERT embeddings")
    ap.add_argument("--input", required=True, nargs="+", help="Input file(s) or directory(ies). Supports .txt, .jsonl (text field), .csv, or directory of .txt")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--text-column", default="text", help="Column/field name for text (CSV/JSONL)")
    ap.add_argument("--model", default="distilbert-base-uncased", help="HF model name")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--append", action="store_true", help="Append to output if exists")

    args = ap.parse_args(argv)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(device)
    model.eval()

    hidden = getattr(model.config, "hidden_size", None)
    if hidden is None:
        print("Warning: model has no hidden_size; assuming 768")
        hidden = 768
    print(f"Model hidden size (target dim): {hidden}")

    mode = "a" if args.append else "w"
    with out_path.open(mode, encoding="utf-8") as fout:
        for in_arg in args.input:
            p = Path(in_arg)
            iterator = iter_texts_from_path(p, text_column=args.text_column)
            for batch in tqdm(batched(iterator, args.batch_size), desc=f"{p.name}"):
                meta, vecs = process_inputs(batch, tokenizer, model, device, args.max_length)
                assert len(meta) == len(vecs) == len(batch)
                for i in range(len(batch)):
                    rec = {
                        "text": batch[i],
                        "input_ids": meta[i]["input_ids"],
                        "attention_mask": meta[i]["attention_mask"],
                        "target": vecs[i],
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
