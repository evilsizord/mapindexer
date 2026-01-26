from pathlib import Path
import numpy as np
from math import atan2, degrees
from bsp_tool import load_bsp

test_bsp = Path("C:\\repos\\mapindexer\\maps\\processed\\layout_del_1\\maps\\layout_del_1.bsp")


bsp = load_bsp(str(test_bsp))

print("Loaded BSP:", test_bsp)
print("Num Entities:", len(bsp.ENTITIES))
print("Entity0:", bsp.ENTITIES[0])
print("Entity1:", bsp.ENTITIES[1])
print("Entity2:", bsp.ENTITIES[2])
print(bsp.ENTITIES[2].get("classname"))
