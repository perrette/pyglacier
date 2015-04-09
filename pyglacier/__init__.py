import warnings

try:
    from .version import version as __version__
except ImportError as error:
    warnings.warn(error.message)
    __version__ = "unknown"

import settings
from settings import set_paths

def setup(*args, **kwargs):
    global Glacier
    set_paths(*args, **kwargs)
    print "==============="
    print "Setup pyglacier"
    print "==============="
    print "...pyglacier version: ", __version__
    print "...code directory: ",settings.CODEDIR
    from pyglacier.run import get_checksum
    print "...fortran code version: ", get_checksum(warn_if_dirty=True).strip()
    print "...default executable: ",settings.EXE
    print "...default glacier lib: ",settings.GLACIERLIB
    print "...default namelist: ",settings.NML
    print "...default output directory: ",settings.OUTDIR
    from pyglacier.core import Glacier
