""" General pyglacier settings
"""
import os

# CODEDIR = os.path.dirname(__file__)+'/../'
# Assume the program is started from the code directory, unless otherwise indicated
CODEDIR = './'
OUTDIR = './out/'  # out directory in calling directory
INFILE = "./input/glacier.nc"
RSTFILE = ""  # not restart file by default

def set_paths(codedir=None, outdir=None, infile=None, rstfile=None, nml=None, lib=None, exe=None):
    """ Set paths

    codedir : look for shared lib in codedir/.obj/wrapper.so
        and for default namelist in codedir/params.nml
    """
    global CODEDIR, GLACIERLIB, NML, OUTDIR, INFILE, RSTFILE, EXE
    if codedir is not None:
        CODEDIR = codedir + '/'
    if lib is None:
        GLACIERLIB = CODEDIR +'.obj/wrapper.so'
    else:
        GLACIERLIB = lib
    if nml is None:
        NML = CODEDIR +'params.nml'
    else:
        NML = nml
    if exe is None:
        EXE = CODEDIR + 'main.exe'
    else:
        EXE = exe
    if outdir is not None: OUTDIR=outdir
    if rstfile is not None: RSTFILE=rstfile
    if infile is not None: INFILE=infile

set_paths()
