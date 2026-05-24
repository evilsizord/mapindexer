# AGENTS.md

## Commands

Run all commands from the repo root unless specified.

**analyze pk3s**

Unzips all map .pk3 files from input folder, and writes them to output folder. Parses .arena metadata foreach map, and writes results to local sqlite database.

`python analyze_pk3s.py`

**compute geometry stats**

Compute geometry stats for either a single map, or a folder of maps. Updates database record for the processed map(s) with the computed geometry stats. Also does a 3d flood fill to determine playable areas of the map, and serializes this to a file for later use.

```python
python gen_bsp_stats.py c:\path\to\my.bsp  # compute stats for specific map
python gen_bsp_stats.py c:\path\to\all_maps  # compute stats for multiple maps within a folder
```

**Generate screenshots**

Chooses random camera points within a map (based o nthe 3d flood fill), and then takes screenshots from those selected points.

`python camera_points.py`

**AI tagging from screenshots**

Todo...

## Project Structure

db/             # database schemas and queries
mapindexer/     # main logic
maps/           # working folder to hold maps being processed
scripts/        # helper scripts
tests/          # tests



## Working with BSPs

The bsp_tool module is used to load and get details about bsp maps

`from bsp_tool import load_bsp`

A few things to note:

The world model can be obtained with:
`world_model = bsp.MODELS[0]`

