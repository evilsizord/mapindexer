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
