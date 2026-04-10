#!/usr/bin/env python3
"""Reference embeddings (Hugging Face transformers + CUDA) matching `embed-in-process embed` YAML schema."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch
import yaml
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="nvidia/llama-nv-embed-reasoning-3b")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-yaml", type=Path, required=True)
    p.add_argument("--max-chunk-bytes", type=int, default=8192)
    p.add_argument("--query-prefix", default="query: ")
    p.add_argument("--passage-prefix", default="passage: ")
    args = p.parse_args()

    input_dir = args.input_dir.resolve()
    query_dir = input_dir / "query"
    passage_dir = input_dir / "passage"
    if not query_dir.is_dir():
        raise SystemExit(f"missing {query_dir}")
    if not passage_dir.is_dir():
        raise SystemExit(f"missing {passage_dir}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    model.cuda()

    entries: list[dict] = []
    for task, sub, prefix in [
        ("query", "query", args.query_prefix),
        ("passage", "passage", args.passage_prefix),
    ]:
        d = input_dir / sub
        for name in sorted(os.listdir(d)):
            path = d / name
            if not path.is_file():
                continue
            data = path.read_bytes()
            if len(data) > args.max_chunk_bytes:
                raise SystemExit(
                    f"{path}: {len(data)} bytes > --max-chunk-bytes {args.max_chunk_bytes}"
                )
            text = data.decode("utf-8", errors="replace")
            full = prefix + text
            inputs = tok(
                full,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=8192,
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                hidden = out.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-12)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            vec = pooled[0].float().cpu().numpy().tolist()
            rel = f"{sub}/{name}".replace("\\", "/")
            sha = hashlib.sha256(data).hexdigest()
            entries.append(
                {
                    "path": rel,
                    "task": task,
                    "byte_length": len(data),
                    "sha256": sha,
                    "embedding": vec,
                }
            )

    out_doc = {
        "model": args.model,
        "engine": "transformers_cuda",
        "entries": entries,
    }
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with args.output_yaml.open("w") as f:
        yaml.safe_dump(out_doc, f, default_flow_style=False, sort_keys=False)
    print(f"wrote {args.output_yaml}")


if __name__ == "__main__":
    main()
