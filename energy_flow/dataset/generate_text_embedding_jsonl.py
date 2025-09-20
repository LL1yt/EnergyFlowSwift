#!/usr/bin/env python3
"""
Unified embedding generator leveraging energy_flow.dataset infrastructure.

Now supports two output modes:
1) JSONL (one JSON object per line):
   {
       "text": "...",
       "input_ids": [int, ...],        # token ids (optional)
       "attention_mask": [0/1, ...],   # mask (optional)
       "target": [float, ...]          # normalized embedding vector (teacher model)
   }

2) Compact binary EFB (.efb) format (Swift-friendly, macOS-friendly):
   - Little-endian layout, no external deps needed in Swift
   - File layout:
       magic: 4 bytes = b"EFB1"
       num_samples: uint32
       embedding_dim: uint32
       repeat num_samples times:
           L: uint32                       # token length
           input_ids: L x uint32
           attention_mask: L x uint8
           target: embedding_dim x float32
   - Contains tokens + masks + target embeddings only (no raw text)

Key differences vs previous standalone script:
    - Uses TeacherModelProvider (same as training pipeline) instead of direct HF calls.
    - Respects DatasetConfig normalization / validation flags.
    - Can force local cache download with --download-local.
    - Provides --no-normalize to bypass l2 normalization.
    - Embedding dimension inferred from teacher model (default DistilBERT = 768).

Supported inputs:
    * .txt (one text per line)
    * .jsonl (expects field with text, default: 'text')
    * .csv (requires pandas, column name configurable)
    * Directory of .txt files

Examples:
    # JSONL only
    python generate_text_embedding_jsonl.py --input texts.txt --output out.jsonl
    # JSONL disabled, EFB only
    python generate_text_embedding_jsonl.py --input data.jsonl --text-column text --no-jsonl --efb-output out.efb
    # Both JSONL and EFB
    python generate_text_embedding_jsonl.py --input dir_with_txt --batch-size 64 --output out.jsonl --efb-output out.efb

Notes:
    - CUDA is used if available (same policy as dataset module).
    - For very large corpora, prefer splitting input for memory locality.
    - Use --append to continue writing to an existing JSONL file.
    - Use --progress-interval to reduce tqdm refresh cost in massive runs.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional
import struct
import array

import torch
from tqdm import tqdm

# Reuse project infrastructure
from .config import DatasetConfig
from .providers import create_teacher_model_provider

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


def _masked_mean(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Local masked mean (fallback) -- normally TeacherModelProvider already does mean pooling.
    This is ONLY used if we directly access model outputs (e.g., --raw-forward flag in future).
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def encode_with_provider(provider, texts: List[str], max_length: int, return_tokens: bool = True) -> Tuple[List[Dict[str, List[int]]], List[List[float]]]:
    """Encode texts using TeacherModelProvider (leveraging its cache and normalization).

    We temporarily access its tokenizer / model to reproduce token ids & attention masks for JSONL/EFB.
    Embeddings are retrieved via provider.encode_texts (normalized if config.normalize_embeddings=True).
    """
    if not texts:
        return [], []

    tokenizer = provider.tokenizer
    model = provider.model
    device = provider.device

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
        pooled = _masked_mean(outputs.last_hidden_state, enc["attention_mask"])  # [B,H]

    if provider.config.normalize_embeddings:
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

    pooled_cpu = pooled.to("cpu")
    input_ids_cpu = enc["input_ids"].to("cpu")
    attention_cpu = enc["attention_mask"].to("cpu")

    meta: List[Dict[str, List[int]]] = []
    if return_tokens:
        for i in range(pooled_cpu.size(0)):
            meta.append({
                "input_ids": input_ids_cpu[i].tolist(),
                "attention_mask": attention_cpu[i].tolist(),
            })
    else:
        meta = [{} for _ in range(pooled_cpu.size(0))]

    return meta, pooled_cpu.tolist()


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Generate embeddings via energy_flow dataset infra (JSONL and/or EFB)")
    ap.add_argument("--input", nargs="+", help="Input file(s) or directory(ies). Supports .txt, .jsonl, .csv, or directory of .txt (omit with --from-snli)")
    ap.add_argument("--output", help="Output JSONL path (optional if --no-jsonl or when only --efb-output is used)")
    ap.add_argument("--text-column", default="text", help="Column/field name for text (CSV/JSONL)")
    ap.add_argument("--model", default="distilbert-base-uncased", help="Teacher HF model name (will integrate with local cache)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--append", action="store_true", help="Append to output if exists (JSONL only)")
    ap.add_argument("--no-normalize", action="store_true", help="Disable l2 normalization (overrides DatasetConfig)")
    ap.add_argument("--download-local", action="store_true", help="Force download & use local cached copy of model")
    ap.add_argument("--progress-interval", type=int, default=1, help="tqdm update interval (set higher for huge corpora)")
    ap.add_argument("--no-tokens", action="store_true", help="Skip saving token ids and attention mask (only text + target)")
    ap.add_argument("--pt-output", help="Optional .pt file to save stacked embeddings (and optionally texts)")
    ap.add_argument("--no-jsonl", action="store_true", help="Do not write JSONL (use with --efb-output or --pt-output)")
    ap.add_argument("--efb-output", help="Optional path to write compact binary EFB dataset (.efb). Requires tokens.")
    ap.add_argument("--shard-size", type=int, default=0, help="If >0, create sharded pt outputs with this many samples per shard")
    ap.add_argument("--pack-dataset", action="store_true", help="Save .pt in unified training format (input_embeddings=target_embeddings, text_pairs)")
    ap.add_argument("--save-texts-in-pt", action="store_true", help="Include raw texts list in .pt file")
    # SNLI integration
    ap.add_argument("--from-snli", action="store_true", help="Generate embeddings from SNLI dataset instead of raw input files")
    ap.add_argument("--snli-limit", type=int, default=None, help="Limit number of SNLI pairs (after fraction) to process")
    # Duplicate handling
    ap.add_argument("--dedup", action="store_true", help="Skip duplicate texts (exact match or normalized if --dedup-normalize)")
    ap.add_argument("--dedup-normalize", action="store_true", help="Apply lowercase+strip normalization before duplicate check")
    # Embedding role control when packing dataset (SNLI only): both / input-only / target-only
    ap.add_argument("--pack-role", choices=["both", "input", "target"], default="both", help="When --pack-dataset with --from-snli: which side(s) to store as embeddings")

    args = ap.parse_args(argv)

    # Validate argument combinations
    if not args.from_snli:
        if not args.input:
            ap.error("--input is required unless --from-snli is used")
        # Ensure at least one output is selected
        if (not args.output and not args.pt_output and not args.efb_output) or (not args.output and not args.no_jsonl and not args.efb_output and not args.pt_output):
            # If user didn't disable JSONL and didn't provide output anywhere
            if not args.no_jsonl and not args.output and not args.efb_output and not args.pt_output:
                ap.error("Provide --output for JSONL or use --no-jsonl with --efb-output/--pt-output")
    else:
        # SNLI mode
        if not args.output and not args.no_jsonl:
            # Provide default output if user forgot and wants JSONL
            args.output = "snli_embeddings.jsonl"
        if not args.pt_output:
            # Provide a default pt output if pack-dataset requested
            if args.pack_dataset:
                args.pt_output = "snli_pack.pt"
            elif args.no_jsonl:
                # Flatten embeddings only
                args.pt_output = "snli_embeddings.pt"

    if args.efb_output and args.no_tokens:
        ap.error("--efb-output requires tokens. Do not pass --no-tokens.")
    if args.efb_output and args.from_snli and args.pack_dataset:
        ap.error("--efb-output is not supported with --from-snli --pack-dataset (no tokens in that fast path). Use SNLI flat mode or disable efb.")

    # Build DatasetConfig tuned for embedding extraction
    ds_config = DatasetConfig(
        teacher_model=args.model,
        use_local_model=args.download_local or True,  # keep local preference
        normalize_embeddings=not args.no_normalize,
        batch_size=args.batch_size,
        dataset_sources=["precomputed"],  # irrelevant here but required
        validate_embeddings=False,  # speed
    )

    provider = create_teacher_model_provider(ds_config)
    if not provider.ensure_initialized():
        print("Failed to initialize teacher model provider", file=sys.stderr)
        return 1

    hidden = getattr(provider.model.config, "hidden_size", 768)
    print(f"Teacher model ready: {args.model} (hidden={hidden}) on {provider.device}")

    # EFB writer (compact binary)
    efb_writer: Optional[EFBBinaryWriter] = None
    if args.efb_output:
        efb_path = Path(args.efb_output)
        efb_path.parent.mkdir(parents=True, exist_ok=True)
        efb_writer = EFBBinaryWriter(str(efb_path), embed_dim=hidden)

    # Prepare JSONL writer if needed
    out_path = Path(args.output) if args.output else Path("snli_embeddings.jsonl")
    if not args.no_jsonl:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    # Accumulators for optional .pt saving
    all_embeddings: List[torch.Tensor] = []
    all_texts: List[str] = []
    shard_index = 0
    shard_written = 0

    def flush_shard(force: bool = False):
        nonlocal shard_index, shard_written, all_embeddings, all_texts
        if args.shard_size <= 0:
            return
        # Flush when shard is full or forced
        if (args.shard_size > 0 and shard_written >= args.shard_size) or (force and shard_written > 0):
            shard_embeddings = torch.cat(all_embeddings, dim=0)
            if args.pack_dataset:
                save_obj = {
                    'input_embeddings': shard_embeddings,
                    'target_embeddings': shard_embeddings.clone(),
                    'generation_info': {
                        'mode': 'embedding_only',
                        'actual_pairs': shard_embeddings.shape[0],
                        'embedding_dimension': shard_embeddings.shape[1]
                    }
                }
                if args.save_texts_in_pt:
                    save_obj['text_pairs'] = [(t, t) for t in all_texts]
            else:
                save_obj = {
                    'embeddings': shard_embeddings
                }
                if args.save_texts_in_pt:
                    save_obj['texts'] = list(all_texts)
            base_stem = Path(args.pt_output).stem if args.pt_output else out_path.stem
            base_dir = Path(args.pt_output).parent if args.pt_output else out_path.parent
            shard_name = f"{base_stem}_shard{shard_index:04d}.pt"
            shard_path = base_dir / shard_name
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(save_obj, shard_path)
            print(f"[shard] saved {shard_path} ({shard_embeddings.shape[0]} samples)")
            shard_index += 1
            shard_written = 0
            all_embeddings.clear()
            all_texts.clear()

    jsonl_file = None
    if not args.no_jsonl:
        jsonl_file = out_path.open(mode, encoding="utf-8")

    total_written = 0
    dedup_set = set()

    def normalize_text(t: str) -> str:
        return t.strip().lower() if args.dedup_normalize else t

    # Optional SNLI mode: bypass raw file iteration
    if args.from_snli:
        # Lazy import to avoid circulars
        from .config import DatasetConfig as _DC
        from .manager import create_dataset_manager
        from .providers import create_snli_provider
        snli_cfg = _DC(
            teacher_model=args.model,
            dataset_sources=["snli"],
            snli_fraction=ds_config.snli_fraction,
            normalize_embeddings=ds_config.normalize_embeddings,
            validate_embeddings=False,
            batch_size=args.batch_size
        )
        # Reuse existing teacher provider to avoid double loading
        snli_provider = create_snli_provider(snli_cfg, provider)
        if not snli_provider.ensure_initialized():
            print("Failed to initialize SNLI provider", file=sys.stderr)
            return 1
        # Get pairs (premise, hypothesis)
        pairs = snli_provider.get_text_pairs(max_samples=args.snli_limit)
        if not pairs:
            print("No SNLI pairs retrieved", file=sys.stderr)
            return 1
        # Flatten into single text list (we treat each side independently for embedding export unless packing dataset)
        # For packing dataset we keep original structure.
        if args.pack_dataset:
            input_texts = [p[0] for p in pairs]
            target_texts = [p[1] for p in pairs]
            # Generate embeddings only for needed roles
            if args.pack_role in ("both", "input"):
                input_emb = provider.encode_texts(input_texts).cpu()
            else:
                input_emb = provider.encode_texts(["" for _ in input_texts]) * 0  # dummy zeros (will be ignored)
            if args.pack_role in ("both", "target"):
                target_emb = provider.encode_texts(target_texts).cpu()
            else:
                target_emb = provider.encode_texts(["" for _ in target_texts]) * 0
            if args.pt_output:
                pack_obj = {
                    'input_embeddings': input_emb if args.pack_role != 'target' else target_emb.clone(),
                    'target_embeddings': target_emb if args.pack_role != 'input' else input_emb.clone(),
                    'text_pairs': list(pairs),
                    'generation_info': {
                        'mode': 'snli_pack',
                        'pairs': len(pairs),
                        'pack_role': args.pack_role
                    }
                }
                pt_path = Path(args.pt_output)
                pt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(pack_obj, pt_path)
                print(f"Saved SNLI packed dataset: {pt_path} ({len(pairs)} pairs)")
            if not args.no_jsonl and jsonl_file is not None:
                for i, (a, b) in enumerate(pairs):
                    rec = {"text": a, "target": input_emb[i].tolist(), "pair_target": b}
                    jsonl_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_written += 1
            if jsonl_file is not None:
                jsonl_file.close()
            if args.pt_output and args.shard_size > 0:
                print("Warning: shard_size ignored in SNLI pack mode")
            if efb_writer is not None:
                efb_writer.close()
            return 0
        else:
            flat_texts: List[str] = []
            for a, b in pairs:
                for t in (a, b):
                    key = normalize_text(t) if args.dedup else None
                    if args.dedup and key in dedup_set:
                        continue
                    if args.dedup and key is not None:
                        dedup_set.add(key)
                    flat_texts.append(t)
            # Batch over flat_texts using existing loop logic by simulating one input source
            iterator = iter(flat_texts)
            # Reuse rest of logic by constructing artificial batch generator below
            for batch in tqdm(batched(iterator, args.batch_size), desc="SNLI", mininterval=args.progress_interval):
                meta, vecs = encode_with_provider(provider, batch, max_length=args.max_length, return_tokens=(not args.no_tokens) or (efb_writer is not None))
                if efb_writer is not None:
                    for i in range(len(batch)):
                        if not meta or not meta[i]:
                            print("EFB writer requires tokens; got none", file=sys.stderr)
                            return 1
                        efb_writer.write_sample(meta[i]["input_ids"], meta[i]["attention_mask"], vecs[i])
                if args.pt_output or args.shard_size > 0:
                    all_embeddings.append(torch.tensor(vecs, dtype=torch.float32))
                    all_texts.extend(batch)
                    shard_written += len(batch)
                    flush_shard(force=False)
                if jsonl_file is not None and not args.no_jsonl:
                    for i, text in enumerate(batch):
                        rec = {"text": text, "target": vecs[i]}
                        if meta and meta[i]:
                            rec.update({
                                "input_ids": meta[i].get("input_ids", []),
                                "attention_mask": meta[i].get("attention_mask", []),
                            })
                        jsonl_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_written += 1
            # After SNLI mode finish standard finalization below
            if jsonl_file is not None:
                jsonl_file.close()
            if args.shard_size > 0:
                flush_shard(force=True)
            if (args.pt_output and args.shard_size <= 0) and all_embeddings:
                embeddings_tensor = torch.cat(all_embeddings, dim=0)
                pt_obj = {'embeddings': embeddings_tensor}
                if args.save_texts_in_pt:
                    pt_obj['texts'] = all_texts
                pt_path = Path(args.pt_output)
                pt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(pt_obj, pt_path)
                print(f"Saved SNLI embeddings tensor: {pt_path} ({embeddings_tensor.shape[0]} samples)")
            if not args.no_jsonl:
                print(f"Done. Wrote {total_written} JSONL SNLI records")
            if efb_writer is not None:
                efb_writer.close()
            return 0
    try:
        for in_arg in args.input:
            p = Path(in_arg)
            iterator = iter_texts_from_path(p, text_column=args.text_column)
            for batch in tqdm(batched(iterator, args.batch_size), desc=f"{p.name}", mininterval=args.progress_interval):
                # Deduplicate inside batch
                if args.dedup:
                    filtered_batch = []
                    for t in batch:
                        key = normalize_text(t)
                        if key in dedup_set:
                            continue
                        dedup_set.add(key)
                        filtered_batch.append(t)
                    batch = filtered_batch
                    if not batch:
                        continue
                meta, vecs = encode_with_provider(provider, batch, max_length=args.max_length, return_tokens=(not args.no_tokens) or (efb_writer is not None))
                # EFB write
                if efb_writer is not None:
                    for i in range(len(batch)):
                        if not meta or not meta[i]:
                            print("EFB writer requires tokens; got none", file=sys.stderr)
                            return 1
                        efb_writer.write_sample(meta[i]["input_ids"], meta[i]["attention_mask"], vecs[i])
                # Accumulate embeddings for .pt
                if args.pt_output or args.shard_size > 0:
                    all_embeddings.append(torch.tensor(vecs, dtype=torch.float32))
                    all_texts.extend(batch)
                    shard_written += len(batch)
                    flush_shard(force=False)
                # Write JSONL
                if jsonl_file is not None:
                    for i, text in enumerate(batch):
                        rec = {"text": text, "target": vecs[i]}
                        if meta and meta[i]:
                            rec.update({
                                "input_ids": meta[i].get("input_ids", []),
                                "attention_mask": meta[i].get("attention_mask", []),
                            })
                        jsonl_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_written += 1
    finally:
        if jsonl_file is not None:
            jsonl_file.close()
        # Force flush last shard
        if args.shard_size > 0:
            flush_shard(force=True)
        if efb_writer is not None:
            efb_writer.close()

    # Write single .pt if requested and not sharded
    if (args.pt_output and args.shard_size <= 0) and all_embeddings:
        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        if args.pack_dataset:
            pt_obj = {
                'input_embeddings': embeddings_tensor,
                'target_embeddings': embeddings_tensor.clone(),
                'generation_info': {
                    'mode': 'embedding_only',
                    'actual_pairs': embeddings_tensor.shape[0],
                    'embedding_dimension': embeddings_tensor.shape[1]
                }
            }
            if args.save_texts_in_pt:
                pt_obj['text_pairs'] = [(t, t) for t in all_texts]
        else:
            pt_obj = {'embeddings': embeddings_tensor}
            if args.save_texts_in_pt:
                pt_obj['texts'] = all_texts
        pt_path = Path(args.pt_output)
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pt_obj, pt_path)
        print(f"Saved embeddings tensor: {pt_path} ({embeddings_tensor.shape[0]} samples)")

    if not args.no_jsonl:
        print(f"Done. Wrote {total_written} JSONL records to: {out_path}")
    return 0


class EFBBinaryWriter:
    def __init__(self, path: str, embed_dim: int):
        self.path = path
        self.embed_dim = int(embed_dim)
        self.f = open(path, 'wb+')
        # Write header with placeholder for num_samples
        self.f.write(b'EFB1')
        self.f.write(struct.pack('<I', 0))  # num_samples placeholder
        self.f.write(struct.pack('<I', self.embed_dim))
        self.count = 0

    def write_sample(self, input_ids: List[int], attention_mask: List[int], target: List[float]):
        L = int(len(input_ids))
        if len(attention_mask) != L:
            raise ValueError('attention_mask length != input_ids length')
        if len(target) != self.embed_dim:
            raise ValueError('target embedding dimension mismatch')
        # token length
        self.f.write(struct.pack('<I', L))
        # input_ids as uint32
        ids_arr = array.array('I', (int(x) for x in input_ids))
        self.f.write(ids_arr.tobytes())
        # attention_mask as uint8
        mask_arr = array.array('B', (1 if int(x) != 0 else 0 for x in attention_mask))
        self.f.write(mask_arr.tobytes())
        # target as float32
        tgt_arr = array.array('f', (float(x) for x in target))
        self.f.write(tgt_arr.tobytes())
        self.count += 1

    def close(self):
        if self.f is None:
            return
        # Patch num_samples in header
        self.f.flush()
        self.f.seek(4)
        self.f.write(struct.pack('<I', self.count))
        self.f.flush()
        self.f.close()
        self.f = None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
