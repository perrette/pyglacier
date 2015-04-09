""" General pyglacier settings
"""
import os

# CODEDIR = os.path.dirname(__file__)+'/../'
# Assume the program is started from the code directory, unless otherwise indicated

def set_codedir(codedir):
    global CODEDIR, GLACIERLIB, NML, OUTDIR, INFILE, RSTFILE
    CODEDIR = codedir + '/'
    GLACIERLIB = CODEDIR +'.obj/wrapper.so'
    NML = CODEDIR +'params.nml'
    OUTDIR = './out/'  # out directory in calling directory
    INFILE = "./input/glacier.nc"
    RSTFILE = ""  # not restart file by default

set_codedir('./')
