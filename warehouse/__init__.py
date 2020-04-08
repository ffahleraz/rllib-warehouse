from . import default
from .default import *

from . import hard
from .hard import *

__all__ = ["default", "hard"]
__all__.extend(default.__all__)
__all__.extend(hard.__all__)

name = "warehouse"
