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