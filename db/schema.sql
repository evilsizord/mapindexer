CREATE TABLE Maps (
    id INTEGER PRIMARY KEY,         -- sqlite recommends not to use autoincrement keyword, but it should still autoincrmeent
    name TEXT NOT NULL,
    arena_longname TEXT,
    author TEXT,
    "version" TEXT,
    release_date TEXT,              -- ISO-8601: YYYY-MM-DD
    verticality_score REAL,          -- normalized 0.0 - 1.0
    complexity_score REAL,            -- normalized 0.0 - 1.0
    "file_name" TEXT, 
    "play_size" TEXT, 
    "file_size" INTEGER, 
    "supported_modes" TEXT
);

CREATE TABLE MapScreenshots (
    id INTEGER PRIMARY KEY,
    map_id INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    camera_type TEXT,                -- e.g. overview, combat, sniper

    FOREIGN KEY (map_id)
        REFERENCES Maps (id)
        ON DELETE CASCADE
);

CREATE TABLE MapTags (
    id INTEGER PRIMARY KEY,
    map_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),

    FOREIGN KEY (map_id)
        REFERENCES Maps (id)
        ON DELETE CASCADE,

    UNIQUE (map_id, tag)
);

CREATE TABLE MapGameModes (
    id INTEGER PRIMARY KEY,
    map_id INTEGER NOT NULL,
    mode TEXT NOT NULL,              -- e.g. CTF, TS, Bomb

    FOREIGN KEY (map_id)
        REFERENCES Maps (id)
        ON DELETE CASCADE,

    UNIQUE (map_id, mode)
);

CREATE INDEX idx_maps_name
    ON Maps (name);
CREATE INDEX idx_maps_author
    ON Maps (author);
CREATE INDEX idx_screenshots_map_id
    ON MapScreenshots (map_id);
CREATE INDEX idx_tags_tag
    ON MapTags (tag);
CREATE INDEX idx_tags_map_id
    ON MapTags (map_id);
CREATE INDEX idx_modes_map_id
