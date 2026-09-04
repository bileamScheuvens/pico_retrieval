from pathlib import Path

# relative paths
ROOT = Path(__file__).parent.parent
CACHEPATH = ROOT / ".cache"
INDEXPATH = ROOT / ".index"
SHARDPATH = INDEXPATH / "shards"
EVALPATH = ROOT / "eval"
CONFIGPATH = ROOT / "conf"
EXPORTPATH = ROOT / "exports"

# global paths, set as desired
MODELPATH = Path.home() / "models"
DATAPATH = Path.home() / "data"
