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

.. that can be used with mapviewer


# Todo

 - consider upgrading to qwen3
 - sqlite is probably not scalable, what is replacement?
 

