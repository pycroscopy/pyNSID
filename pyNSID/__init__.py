from . import io
from .io import *
from .__version__ import version as __version__

__all__ = ['__version__', 'io']
__all__ += io.__all__
