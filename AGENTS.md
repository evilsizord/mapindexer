# AGENTS.md

## Commands

Run from the repo root unless specified.



## Project Structure

db/             # database schemas and queries
mapindexer/     # main logic
maps/           # working folder to hold maps being processed
scripts/        # helper scripts
tests/          # tests


## Conventions




## Working with BSPs

The bsp_tool module is used to load and get details about bsp maps

`from bsp_tool import load_bsp`

A few things to note:

The world model can be obtained with:
`world_model = bsp.MODELS[0]`

