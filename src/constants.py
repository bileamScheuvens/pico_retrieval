from pathlib import Path

ROOT = Path(__file__).parent.parent
CACHEPATH = ROOT / ".cache"
INDEXPATH = ROOT / ".index"
SHARDPATH = INDEXPATH / "shards"
EVALPATH = ROOT / "eval"
MODELPATH = Path.home() / "models"
DATAPATH = Path.home() / "data"
CONFIGPATH = ROOT / "conf"
