# path hacking
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.resolve() / "scirepeval"))

from .scirepeval.scirepeval import SciRepEval
from .scirepeval.evaluation.encoders import Model as SciRepModel
