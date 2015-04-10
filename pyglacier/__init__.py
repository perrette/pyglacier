from __future__ import print_function, absolute_import
import sys, warnings

try:
    from .version import version as __version__
except ImportError as error:
    warnings.warn(error.message)
    __version__ = "unknown"

from .  import settings
from .settings import set_paths

def setup(*args, **kwargs):
    global Glacier
    set_paths(*args, **kwargs)
    print_info()
    try:
        import pyglacier.wrapper
        pyglacier.wrapper.load_lib(settings.GLACIERLIB)
    except Exception as error:
        warnings.warn(error.message)
    from pyglacier.core import Glacier

def print_info():
    print("===============")
    print("Setup pyglacier")
    print("===============")
    print("...pyglacier version: ", __version__)
    print("...fortran code directory: ",settings.CODEDIR)
    from pyglacier.run import get_checksum
    print("...fortran code version: ", get_checksum(warn_if_dirty=True).strip()[:7])
    print("...default executable: ",settings.EXE.replace(settings.CODEDIR, "$CODEDIR/"))
    print("...default glacier lib: ",settings.GLACIERLIB.replace(settings.CODEDIR, "$CODEDIR/"))
    print("...default namelist: ",settings.NML.replace(settings.CODEDIR, "$CODEDIR/"))
    print("...default output directory: ",settings.OUTDIR.replace(settings.CODEDIR, "$CODEDIR/"))
