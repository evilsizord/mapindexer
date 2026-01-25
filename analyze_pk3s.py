import os
from pathlib import Path
from mapindexer.extract_pk3 import extract_pk3
from dotenv import load_dotenv
import sqlite3

load_dotenv()

RAW = Path(os.getenv("MAPS_RAW_DIR"))
OUT = Path(os.getenv("MAPS_OUT_DIR"))
db_path = os.getenv("MAPINDEXER_DB")

if not db_path:
    raise SystemExit("MAPINDEXER_DB environment variable not set")

try:
    dbconn = sqlite3.connect(db_path)
except sqlite3.Error as e:
    raise SystemExit(f"Failed to connect to database {db_path}: {e}")

for pk3 in RAW.glob("*.pk3"):
    try:
        extract_pk3(pk3, OUT, db_conn=dbconn)
        print(f"Extracted {pk3.name}")
    except Exception as e:
        print(f"Failed {pk3.name}: {e}")
