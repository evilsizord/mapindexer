# AI Agent Conversations

Below is a selection of some relevent AI chats used in this project. 

https://chatgpt.com/share/69697f02-215c-8009-a2ce-fcfae2fc14de

---

> I want to create a massive database of maps (levels) for the game Urban terror. The database should have keywords describing each map, and should have the author of the map and screenshots of each map. Is there a way to use AI to somehow generate that database from the map files?

Yes — you *can* use AI to generate most of that database automatically from Urban Terror map files, but it will work best as a **hybrid pipeline** (deterministic extraction + AI enrichment), not “AI-only.”

Below is a **practical, scalable architecture** that many game studios and mod communities use for exactly this kind of problem.

---

## 1. What’s inside an Urban Terror map (and what you can extract without AI)

Urban Terror maps are typically distributed as **`.pk3` files** (zip archives).

Inside you’ll usually find:

* **`.bsp`** → compiled level geometry
* **`.arena`** → metadata (map name, author, supported game types)
* **`.shader` / `.tga` / `.jpg`** → textures
* **`.aas`** → bot navigation
* **Levelshot** → screenshot used in the map menu

### Automatically extractable (no AI needed)

You can reliably extract:

* ✅ Map name
* ✅ Author(s)
* ✅ Supported game modes
* ✅ Existing levelshot (if present)
* ✅ File size, release date, version (from headers)

**Tools**

* Standard zip tools for `.pk3`
* Python + `zipfile`
* Urban Terror / Quake 3 BSP parsers (e.g. `q3bsp`, `bsp-tool`)

This gives you **ground truth metadata** — AI should *never* guess this if it can be parsed.

---

## 2. Generating screenshots automatically (critical step)

Many maps either:

* Have no levelshot
* Have a poor-quality or outdated one

### Best approach

Run Urban Terror (or ioquake3) **headless or scripted** and:

1. Load the map
2. Teleport to predefined camera points
3. Capture multiple screenshots
4. Save at consistent resolution

This is how map repositories like **MapHub** or **Steam Workshop pipelines** work.

**Tools**

* Autoexec scripts
* Camera path scripts
* `screenshotJPEG`
* Docker + virtual GPU (for scale)

📌 *This step dramatically improves AI tagging accuracy later.*

---

## 3. AI-generated keywords & descriptions (where AI shines)

Once you have **screenshots + basic metadata**, AI becomes extremely effective.

### What AI can generate reliably

From screenshots + optional BSP stats:

* Keywords:

  * `small`, `medium`, `large`
  * `urban`, `industrial`, `night`, `snow`
  * `vertical`, `open`, `tight corridors`
  * `sniper-friendly`, `close-quarters`
* Gameplay tags:

  * `CTF-focused`
  * `bomb-heavy`
  * `balanced`, `defender-advantage`
* Visual style:

  * `realistic`, `gritty`, `clean`, `dark`
* Complexity estimates:

  * `high navigation complexity`
  * `simple layout`

### How to do it

Use a **vision-capable model** on:

* 3–6 screenshots per map
* Optional: top-down layout image (if you generate one)

Example prompt idea:

> “Analyze these screenshots from an Urban Terror map. Generate 10–20 concise gameplay and visual keywords. Avoid guessing author or name.”

---

## 4. Optional: Extracting layout intelligence from BSP (advanced)

This is optional but powerful.

From the `.bsp` you can compute:

* Map bounding box size
* Vertical variance (Z-axis range)
* Number of rooms / vis clusters
* Average corridor width
* Navmesh density (from `.aas`)

These metrics can:

* Feed into AI prompts
* Auto-classify maps (`small / medium / large`)
* Power advanced search filters

---

## 5. Suggested database schema

```sql
Maps
- id
- name
- author
- version
- release_date
- file_hash
- bsp_size
- verticality_score
- complexity_score

MapScreenshots
- map_id
- image_path
- camera_type

MapTags
- map_id
- tag
- confidence

MapGameModes
- map_id
- mode
```

AI-generated tags should always be **confidence-weighted**, not absolute.

---

## 6. Human-in-the-loop (important)

Even the best AI will:

* Mislabel rare or experimental maps
* Miss gameplay nuance

Best practice:

* Allow community moderation
* Let users upvote/downvote tags
* Track AI vs human corrections

This improves trust and long-term quality.

---

## 7. What AI *cannot* do reliably (yet)

🚫 Infer the *intended* gameplay style from geometry alone
🚫 Determine balance quality without match data
🚫 Identify author if missing from metadata
🚫 Replace actual screenshots with “representative art”

---

## 8. Recommended tech stack

**Backend**

* Python (map extraction + orchestration)
* SQLite/Postgres
* Celery / background jobs

**AI**

* Vision-capable LLM for tagging
* Embeddings for similarity search (“maps like Turnpike”)

**Frontend**

* Tag-based search
* Screenshot gallery
* “Maps similar to this” feature (very popular)

---

## 9. Is this worth doing?

Absolutely — Urban Terror lacks a **modern, searchable map index**, and this approach could become *the* canonical resource for:

* Server admins
* Competitive players
* Modders
* New players

If you want, next I can:

* Design the **exact extraction script**
* Help you craft **AI prompts** for consistent tagging
* Propose a **fully automated pipeline diagram**
* Help you decide whether to open-source it

Just tell me how deep you want to go.

---

> Design the exact extraction script

Below is an **exact, end-to-end extraction script design** you can actually implement. I’ll give you:

1. **Overall pipeline flow**
2. **Concrete file structure**
3. **Exact Python script (single-map + batch)**
4. **Where AI slots in (cleanly separated)**
5. **Next-step extensions**

This is designed so you can run it on **thousands of `.pk3` files** safely and deterministically.

---

# 1. High-level pipeline (deterministic first, AI last)

```
.pk3 file
   ↓
Extract archive
   ↓
Parse .arena metadata
   ↓
Detect .bsp + levelshot
   ↓
Compute geometry stats (optional)
   ↓
Generate screenshots (engine)
   ↓
AI tagging from screenshots
   ↓
Insert into database
```

**Key principle**
👉 *Never let AI guess anything you can extract.*

---

# 2. Directory layout (important for scale)

```
urbanterror-map-db/
│
├── maps_raw/                # Original .pk3 files
│
├── maps_extracted/
│   └── ut4_turnpike/
│       ├── map.pk3
│       ├── arena.json
│       ├── bsp/
│       │   └── ut4_turnpike.bsp
│       ├── levelshots/
│       │   ├── auto_01.jpg
│       │   ├── auto_02.jpg
│       │   └── auto_03.jpg
│       └── stats.json
│
├── scripts/
│   ├── extract_pk3.py
│   ├── parse_arena.py
│   ├── bsp_stats.py
│   ├── capture_screenshots.py
│   ├── ai_tagging.py
│   └── ingest_db.py
│
└── maps.db
```

---

# 3. Core extraction script (exact implementation)

## `extract_pk3.py`

This script:

* Reads `.pk3`
* Finds `.arena`, `.bsp`, existing levelshot
* Outputs **normalized JSON**

```python
import zipfile
import json
import os
from pathlib import Path

def extract_pk3(pk3_path, output_root):
    pk3_path = Path(pk3_path)
    map_name = pk3_path.stem.lower()

    out_dir = Path(output_root) / map_name
    out_dir.mkdir(parents=True, exist_ok=True)

    arena_data = {}
    bsp_files = []
    levelshots = []

    with zipfile.ZipFile(pk3_path, 'r') as z:
        for name in z.namelist():
            lname = name.lower()

            if lname.endswith('.arena'):
                arena_text = z.read(name).decode('utf-8', errors='ignore')
                arena_data = parse_arena(arena_text)

            elif lname.endswith('.bsp'):
                bsp_path = out_dir / 'bsp' / Path(name).name
                bsp_path.parent.mkdir(exist_ok=True)
                bsp_path.write_bytes(z.read(name))
                bsp_files.append(str(bsp_path))

            elif 'levelshots/' in lname and lname.endswith(('.jpg', '.png')):
                ls_path = out_dir / 'levelshots' / Path(name).name
                ls_path.parent.mkdir(exist_ok=True)
                ls_path.write_bytes(z.read(name))
                levelshots.append(str(ls_path))

    meta = {
        "map_name": arena_data.get("map"),
        "author": arena_data.get("author"),
        "gametypes": arena_data.get("type"),
        "bsp_files": bsp_files,
        "existing_levelshots": levelshots
    }

    with open(out_dir / "arena.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def parse_arena(text):
    """
    Very simple Quake3 arena parser
    """
    result = {}
    for line in text.splitlines():
        if '"' in line:
            parts = line.replace('"', '').split()
            if len(parts) >= 2:
                result[parts[0]] = " ".join(parts[1:])
    return result
```

---

## Batch runner

```python
from pathlib import Path
from extract_pk3 import extract_pk3

RAW = Path("maps_raw")
OUT = Path("maps_extracted")

for pk3 in RAW.glob("*.pk3"):
    try:
        extract_pk3(pk3, OUT)
        print(f"Extracted {pk3.name}")
    except Exception as e:
        print(f"Failed {pk3.name}: {e}")
```

---

# 4. BSP geometry stats (optional but powerful)

## `bsp_stats.py`

Uses bounding boxes to estimate scale & verticality.

```python
import struct
import json

def analyze_bsp(bsp_path):
    with open(bsp_path, 'rb') as f:
        data = f.read()

    # Simplified: locate entity lump, parse worldspawn
    # For full precision, use bsp-tool library

    stats = {
        "estimated_size": len(data),
        "verticality": "unknown"
    }

    return stats
```

Later you can add:

* Z-range
* Room count
* Vis clusters
* Navmesh density

---

# 5. Screenshot generation (engine-controlled)

## `capture_screenshots.py`

This **does NOT use AI**.

```bash
ioquake3 +set fs_game q3ut4 \
         +map ut4_turnpike \
         +exec auto_cam.cfg \
         +wait 200 \
         +screenshotJPEG \
         +quit
```

`auto_cam.cfg` example:

```
cg_thirdPerson 1
viewpos 512 1024 256 90
wait 50
screenshotJPEG
viewpos -300 400 180 180
wait 50
screenshotJPEG
```

---

# 6. AI tagging (cleanly isolated)

## `ai_tagging.py`

Inputs:

* `arena.json`
* 3–6 screenshots

Outputs:

```json
{
  "tags": [
    {"tag": "urban", "confidence": 0.92},
    {"tag": "vertical", "confidence": 0.81},
    {"tag": "sniper-friendly", "confidence": 0.76}
  ],
  "summary": "Medium-sized urban map with vertical sightlines..."
}
```

This makes AI **replaceable** later.

---

# 7. Database ingestion

## `ingest_db.py`

```python
import sqlite3
import json

conn = sqlite3.connect("maps.db")

conn.execute("""
CREATE TABLE IF NOT EXISTS maps (
  id INTEGER PRIMARY KEY,
  name TEXT,
  author TEXT,
  gametypes TEXT,
  summary TEXT
)
""")
```

---

# 8. Why this design works

✅ Fully automated
✅ Deterministic metadata
✅ AI only where subjective judgment is needed
✅ Scales to thousands of maps
✅ Community-correctable

---

# 9. Next things I can design for you

If you want, I can next:

* Build the **AI prompt** for perfect tag consistency
* Add **similar-map search via embeddings**
* Create a **web UI schema**
* Optimize **screenshot camera heuristics**
* Design a **Dockerized pipeline**

Just tell me which layer you want next.
