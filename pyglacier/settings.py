""" General pyglacier settings
"""
import os

# CODEDIR = os.path.dirname(__file__)+'/../'
# Assume the program is started from the code directory, unless otherwise indicated
CODEDIR = './'
GLACIERLIB = CODEDIR +'.obj/wrapper.so'
NML = CODEDIR +'params.nml'
OUTDIR = './out/'  # out directory in calling directory
INFILE = "./input/glacier.nc"
RSTFILE = ""  # not restart file by default
