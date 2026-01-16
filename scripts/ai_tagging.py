#!/usr/bin/env python3
"""
Local AI tagging for Urban Terror / Quake 3 maps using:
  - Hugging Face Transformers
  - Qwen/Qwen2.5-VL-7B-Instruct
  - Multi-image prompt (screenshots) + optional stats/arena JSON

Input:  per-map folder with levelshots/*.jpg|png plus optional stats.json / arena.json
Output: ai_tags.json (summary + tags with confidence + reasons)

Requires:
  pip install -U "git+https://github.com/huggingface/transformers" accelerate
  pip install "qwen-vl-utils[decord]==0.0.8"   (or: pip install qwen-vl-utils)
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# -----------------------------
# Tag vocabulary (controlled)
# -----------------------------

VOCABULARY: Dict[str, List[str]] = {
    "Scale": ["small", "medium", "large"],
    "Environment": ["urban", "industrial", "indoor", "outdoor", "mixed", "night", "day", "snow", "desert"],
    "Layout": ["open", "tight", "maze-like", "linear", "hub-based", "symmetrical", "asymmetrical"],
    "Verticality": ["flat", "moderate-vertical", "high-vertical"],
    "Combat": ["close-quarters", "mid-range", "long-sightlines", "sniper-friendly"],
    "Tone": ["dark", "bright", "gritty", "clean", "realistic"],
}
ALLOWED_TAGS = {t for group in VOCABULARY.values() for t in group}

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


# -----------------------------
# Helpers
# -----------------------------

def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def pick_screenshots(levelshots_dir: Path, max_images: int) -> List[Path]:
    if not levelshots_dir.exists():
        return []
    imgs = sorted(
        [p for p in levelshots_dir.iterdir()
         if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]],
        key=lambda p: p.name.lower()
    )
    return imgs[:max_images]

def build_prompt(arena: Optional[Dict[str, Any]], stats: Optional[Dict[str, Any]]) -> str:
    map_name = (arena or {}).get("map_name") or (arena or {}).get("map") or "unknown"
    author = (arena or {}).get("author") or "unknown"
    gametypes = (arena or {}).get("gametypes") or (arena or {}).get("type") or []

    verticality = (stats or {}).get("verticality_score")
    room_count = (stats or {}).get("room_count")
    complexity = (stats or {}).get("complexity_score")

    vocab_lines = [f"{k}: {', '.join(v)}" for k, v in VOCABULARY.items()]
    vocab_text = "\n".join(vocab_lines)

    # We ask for strict JSON output; we'll still validate in code.
    return f"""
You are analyzing representative screenshots from a single Urban Terror (Quake 3) multiplayer FPS map.
Generate descriptive gameplay and visual tags only. Do NOT judge quality. Do NOT guess author intent.
Do NOT claim facts that aren't visible or strongly implied by geometry.

Map metadata (if present):
- name: {map_name}
- author: {author}
- supported_modes: {', '.join(gametypes) if isinstance(gametypes, list) else gametypes}

Map stats (if present; treat as ground truth):
- verticality_score: {verticality}
- room_count: {room_count}
- complexity_score: {complexity}

Instructions:
1) Prefer tags from the provided vocabulary when applicable.
2) Output JSON ONLY (no Markdown).
3) Include:
   - "summary": one neutral sentence
   - "tags": array of objects: {{"tag": string, "confidence": 0..1, "reason": string}}
   - "extra_tags": up to 5 additional tags (strings) if vocabulary is insufficient
4) Avoid duplicate tags. Keep total tags <= 20.

Vocabulary:
{vocab_text}

Return format example:
{{
  "summary": "...",
  "tags": [
    {{"tag":"urban","confidence":0.92,"reason":"..."}},
    {{"tag":"mixed","confidence":0.71,"reason":"..."}}
  ],
  "extra_tags": ["..."]
}}
""".strip()

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Qwen will usually output pure JSON if instructed, but this is a safe extractor:
    - Finds the first {...} block and attempts json.loads
    """
    text = text.strip()

    # Fast path: entire output is JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first JSON object block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise RuntimeError(f"No JSON object found in model output:\n{text}")

    blob = m.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON extracted from output.\nExtracted:\n{blob}\n\nFull:\n{text}") from e

def normalize_output(out: Dict[str, Any]) -> Dict[str, Any]:
    summary = str(out.get("summary", "")).strip() or "No summary generated."
    tags_in = out.get("tags", [])
    extra_in = out.get("extra_tags", [])

    extra = []
    if isinstance(extra_in, list):
        extra = [str(x).strip() for x in extra_in if str(x).strip()]
    extra = list(dict.fromkeys(extra))[:5]  # unique, max 5

    cleaned_tags = []
    seen = set()

    if isinstance(tags_in, list):
        for t in tags_in[:25]:
            if not isinstance(t, dict):
                continue
            tag = str(t.get("tag", "")).strip()
            if not tag or tag in seen:
                continue

            conf = t.get("confidence", 0.0)
            try:
                conf = float(conf)
            except Exception:
                conf = 0.0
            conf = max(0.0, min(conf, 1.0))

            reason = str(t.get("reason", "")).strip()
            if not reason:
                reason = "No reason provided."

            # Only keep vocab tags in main tags; push others to extra_tags
            if tag in ALLOWED_TAGS:
                cleaned_tags.append({"tag": tag, "confidence": conf, "reason": reason})
                seen.add(tag)
            else:
                if tag not in extra and tag:
                    extra.append(tag)

    # Safety net
    if not cleaned_tags:
        cleaned_tags = [{"tag": "mixed", "confidence": 0.5, "reason": "Fallback tag when classification is uncertain."}]

    # Cap extra tags
    extra = extra[:5]

    return {"summary": summary, "tags": cleaned_tags[:20], "extra_tags": extra}

def insert_into_sqlite(db_path: Path, map_id: int, result: Dict[str, Any]) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        for t in result["tags"]:
            conn.execute(
                """
                INSERT INTO MapTags (map_id, tag, confidence)
                VALUES (?, ?, ?)
                ON CONFLICT(map_id, tag) DO UPDATE SET confidence=excluded.confidence
                """,
                (map_id, t["tag"], float(t["confidence"])),
            )
        conn.commit()
    finally:
        conn.close()


# -----------------------------
# Model call
# -----------------------------

@torch.inference_mode()
def run_qwen_vl(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    # Apply chat template
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Convert message content to image/video tensors (Qwen helper)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": temperature if temperature > 0.0 else None,
    }
    # Remove None values
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    out_ids = model.generate(**inputs, **gen_kwargs)

    # Trim prompt tokens
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-dir", required=True, help="Path to extracted map folder (contains levelshots/)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    ap.add_argument("--max-images", type=int, default=5, help="How many screenshots to use")
    ap.add_argument("--max-new-tokens", type=int, default=320, help="Generation length")
    ap.add_argument("--temperature", type=float, default=0.2, help="0 disables sampling")
    ap.add_argument("--out", default="ai_tags.json", help="Output filename inside map-dir")
    ap.add_argument("--db", default=None, help="Optional path to maps.db (SQLite) to insert MapTags")
    ap.add_argument("--map-id", type=int, default=None, help="Required if --db is set (Maps.id)")
    ap.add_argument("--min-pixels", type=int, default=None, help="Optional: constrain visual tokens (e.g. 256*28*28)")
    ap.add_argument("--max-pixels", type=int, default=None, help="Optional: constrain visual tokens (e.g. 1280*28*28)")
    args = ap.parse_args()

    map_dir = Path(args.map_dir)
    levelshots_dir = map_dir / "levelshots"
    images = pick_screenshots(levelshots_dir, args.max_images)
    if not images:
        raise SystemExit(f"No screenshots found in {levelshots_dir}")

    arena = read_json_if_exists(map_dir / "arena.json")
    stats = read_json_if_exists(map_dir / "stats.json")
    prompt_text = build_prompt(arena, stats)

    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    processor_kwargs = {}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model, **processor_kwargs)

    # Build message with multiple images + one text instruction
    content: List[Dict[str, Any]] = []
    for p in images:
        # Qwen supports file:// paths in messages for multi-image inference
        content.append({"type": "image", "image": f"file://{p.resolve()}"})
    content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content}]

    raw = run_qwen_vl(
        model=model,
        processor=processor,
        messages=messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    parsed = extract_json_from_text(raw)
    result = normalize_output(parsed)

    out_path = map_dir / args.out
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")

    if args.db:
        if args.map_id is None:
            raise SystemExit("--map-id is required when using --db")
        insert_into_sqlite(Path(args.db), args.map_id, result)
        print(f"Inserted/updated MapTags in DB: {args.db} (map_id={args.map_id})")


if __name__ == "__main__":
    main()

"""
Example Usage:
python ai_tagging_local.py --map-dir maps_extracted/ut4_turnpike --max-images 5

Insert tags into SQLite:
python ai_tagging_local.py --map-dir maps_extracted/ut4_turnpike --db maps.db --map-id 123



"""