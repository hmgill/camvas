
from .core import *
from .output import *
from .weights import *
from .visualization import *
from .scripts import * 


__version__ = "1.0.0"



import pathlib

project_root = pathlib.Path(__file__).parent.absolute()

project_paths = {
    "root" : project_root,
    "output" : pathlib.Path(project_root, "output"),
    "weights" : pathlib.Path(project_root, "output", "weights"),
    "visualization" : pathlib.Path(project_root, "visualization"),
    "scripts" : pathlib.Path(project_root, "scripts")
}

