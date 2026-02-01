# mapindexer

urt map indexer thing. (may work with other bsp maps too idk)

## High-level pipeline 

```
Unzip .pk3 files
   ↓
Parse .arena metadata
   ↓
Compute geometry stats
   ↓
Generate screenshots 
   ↓
AI tagging from screenshots
```

Each phase is run independently, and adds/appends to database records. 

We can keep a flag in the database to determine which phase is completed.

Pay close attention to errors, and adjust as needed.

At the end we should have a database with:

 - map metadata
 - screenshots
 - descriptive keywords

.. that can be used with mapviewer. It would be great also to have a feature vector database, to enable
searching for things like "map that had a car over by the lake, and it was nighttime".


## devlog

(because sometimes i only work on this every 2 weeks, and i'll forget where i left off)

### Jan. 25, 2026

I think I left off here working on debug_voxel_test.py. Trying a new approach for identifying "playable bounds" of a map, because existing methods were not working well enough. Well the world AABB bounds are fine, but I want something more precise than a single AABB. Also the camera placement has been really problematic so far. The voxel_test is currently identifying a few  playable nodes, but doesn't seem like enough, and I'm not sure yet if correct.


## Todo

 - consider upgrading to qwen3
 - sqlite is probably not scalable, what is replacement?
 - bsp_tools does not work at all like the ai thought. Needs a lot of rework for geometry extraction
 - should use ORM. Prism?

