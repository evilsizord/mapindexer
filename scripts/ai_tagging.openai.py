#!/usr/bin/env python3
"""
XX
 - DEPRECATED - USE THE QWEN2.5 VERSION INSTEAD SO CAN RUN LOCALLY
XX

AI tagging for Urban Terror / Quake 3 maps using screenshots + optional BSP stats.

- Input: per-map folder with levelshots/*.jpg (or .png), plus optional stats.json / arena.json
- Output: ai_tags.json (summary + tags with confidence)

Requires:
  pip install openai

Environment:
  export OPENAI_API_KEY="..."
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


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

DEFAULT_MODEL = "gpt-4.1-mini"  # supports image inputs via Responses API :contentReference[oaicite:2]{index=2}


# -----------------------------
# Structured output schema
# -----------------------------

TAG_SCHEMA: Dict[str, Any] = {
    "name": "map_tags",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "tags": {
                "type": "array",
                "minItems": 1,
                "maxItems": 25,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "tag": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reason": {"type": "string"},
                    },
                    "required": ["tag", "confidence", "reason"],
                },
            },
            "extra_tags": {
                "type": "array",
                "minItems": 0,
                "maxItems": 5,
                "items": {"type": "string"},
            },
        },
        "required": ["summary", "tags", "extra_tags"],
    },
    "strict": True,
}


# -----------------------------
# Helpers
# -----------------------------

def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def guess_mime(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    # Fallback (still often works)
    return "application/octet-stream"

def image_to_data_url(p: Path) -> str:
    mime = guess_mime(p)
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def pick_screenshots(levelshots_dir: Path, max_images: int) -> List[Path]:
    if not levelshots_dir.exists():
        return []
    imgs = sorted(
        [p for p in levelshots_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]],
        key=lambda x: x.name.lower(),
    )
    return imgs[:max_images]

def build_prompt(arena: Optional[Dict[str, Any]], stats: Optional[Dict[str, Any]]) -> str:
    map_name = (arena or {}).get("map_name") or (arena or {}).get("map") or "unknown"
    author = (arena or {}).get("author") or "unknown"
    gametypes = (arena or {}).get("gametypes") or (arena or {}).get("type") or []

    verticality = (stats or {}).get("verticality_score")
    room_count = (stats or {}).get("room_count")
    complexity = (stats or {}).get("complexity_score")

    vocab_lines = []
    for k, vals in VOCABULARY.items():
        vocab_lines.append(f"{k}: {', '.join(vals)}")
    vocab_text = "\n".join(vocab_lines)

    # Important: tell the model NOT to judge quality, and to prefer vocabulary tags
    prompt = f"""
You are analyzing representative screenshots from a single Urban Terror (Quake 3) multiplayer FPS map.
Generate descriptive gameplay and visual tags only. Do NOT judge quality. Do NOT guess author intent.

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
2) You may add up to 5 additional descriptive tags in `extra_tags` if the vocabulary is insufficient.
3) Provide a neutral 1-sentence summary.
4) For each tag, include confidence 0.0–1.0 and a short reason grounded in visible evidence.

Vocabulary:
{vocab_text}

Output must match the provided JSON schema exactly.
""".strip()

    return prompt


# -----------------------------
# OpenAI call
# -----------------------------

def tag_map_with_ai(
    client: OpenAI,
    model: str,
    prompt_text: str,
    image_paths: List[Path],
    detail: str = "low",
) -> Dict[str, Any]:
    # Build Responses API input with interleaved text + images :contentReference[oaicite:3]{index=3}
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt_text}]

    for p in image_paths:
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(p),
                "detail": detail,  # "low" saves tokens; "high" if you need fine texture details :contentReference[oaicite:4]{index=4}
            }
        )

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": TAG_SCHEMA["name"],
                "strict": TAG_SCHEMA["strict"],
                "schema": TAG_SCHEMA["schema"],
            }
        },
    )

    # Responses API returns convenient output_text; for json_schema it should be the JSON string :contentReference[oaicite:5]{index=5}
    raw = response.output_text
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model did not return valid JSON. Raw output:\n{raw}") from e

    return parsed


# -----------------------------
# Validation / normalization
# -----------------------------

def normalize_output(out: Dict[str, Any]) -> Dict[str, Any]:
    # Keep only allowed tags in main `tags`, push anything else into extra_tags
    tags = out.get("tags", [])
    extra = set(out.get("extra_tags", []))

    cleaned_tags = []
    for t in tags:
        tag = (t.get("tag") or "").strip()
        conf = float(t.get("confidence", 0.0))
        reason = (t.get("reason") or "").strip()

        if not tag or not reason:
            continue

        if tag in ALLOWED_TAGS:
            cleaned_tags.append({"tag": tag, "confidence": max(0.0, min(conf, 1.0)), "reason": reason})
        else:
            extra.add(tag)

    out["tags"] = cleaned_tags
    out["extra_tags"] = sorted([x for x in extra if x])[:5]
    out["summary"] = (out.get("summary") or "").strip()

    if not out["summary"]:
        out["summary"] = "No summary generated."

    if not out["tags"]:
        # As a safety net, never return empty tags
        out["tags"] = [{"tag": "mixed", "confidence": 0.5, "reason": "Fallback tag when classification is uncertain."}]

    return out


# -----------------------------
# SQLite insertion (optional)
# -----------------------------

def insert_into_sqlite(db_path: Path, map_id: int, result: Dict[str, Any]) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")

        # Insert tags
        for t in result["tags"]:
            conn.execute(
                """
                INSERT INTO MapTags (map_id, tag, confidence)
                VALUES (?, ?, ?)
                ON CONFLICT(map_id, tag) DO UPDATE SET confidence=excluded.confidence
                """,
                (map_id, t["tag"], float(t["confidence"])),
            )

        # Insert modes if you also want to store extras as tags (optional)
        # for extra in result.get("extra_tags", []):
        #     conn.execute(
        #         """
        #         INSERT INTO MapTags (map_id, tag, confidence)
        #         VALUES (?, ?, ?)
        #         ON CONFLICT(map_id, tag) DO UPDATE SET confidence=excluded.confidence
        #         """,
        #         (map_id, extra, 0.55),
        #     )

        conn.commit()
    finally:
        conn.close()


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-dir", required=True, help="Path to extracted map folder (contains levelshots/)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Vision-capable model name")
    ap.add_argument("--max-images", type=int, default=5, help="Number of screenshots to send (default 5)")
    ap.add_argument("--detail", choices=["low", "high", "auto"], default="low", help="Image detail level")
    ap.add_argument("--out", default="ai_tags.json", help="Output JSON filename inside map-dir")
    ap.add_argument("--db", default=None, help="Optional path to maps.db (SQLite) to insert MapTags")
    ap.add_argument("--map-id", type=int, default=None, help="Required if --db is set (Maps.id)")
    args = ap.parse_args()

    map_dir = Path(args.map_dir)
    if not map_dir.exists():
        raise SystemExit(f"Map dir not found: {map_dir}")

    levelshots_dir = map_dir / "levelshots"
    images = pick_screenshots(levelshots_dir, args.max_images)
    if not images:
        raise SystemExit(f"No screenshots found in: {levelshots_dir}")

    arena = read_json_if_exists(map_dir / "arena.json")
    stats = read_json_if_exists(map_dir / "stats.json")

    prompt_text = build_prompt(arena, stats)

    client = OpenAI()
    raw_result = tag_map_with_ai(
        client=client,
        model=args.model,
        prompt_text=prompt_text,
        image_paths=images,
        detail=args.detail,
    )

    result = normalize_output(raw_result)

    out_path = map_dir / args.out
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")

    if args.db:
        if args.map_id is None:
            raise SystemExit("--map-id is required when using --db")
        insert_into_sqlite(Path(args.db), args.map_id, result)
        print(f"Inserted/updated tags in DB: {args.db} (map_id={args.map_id})")


if __name__ == "__main__":
    main()

"""
Example usage:
python ai_tagging.py --map-dir maps_extracted/ut4_turnpike --max-images 5 --detail low

Also insert into SQLite (MapTags) if you already have the Maps.id:
python ai_tagging.py --map-dir maps_extracted/ut4_turnpike --db maps.db --map-id 123

Notes that matter in practice

Use detail=low by default for cost/speed, and only use high if you find the model missing subtle texture/theme cues. The detail knob is part of the image input format.

The Responses API accepts multiple input_image parts alongside text in one request.

json_schema strict structured outputs helps prevent broken JSON and missing fields.
"""