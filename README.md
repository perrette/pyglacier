# pyglacier

Wrapper to run and tune the glacier model

Install
-------
python setup.py install

Getting started
---------------

    import pyglacier
    pyglacier.setup('/path/to/glacier/code')

The following output will be printed to screen:

===============
Setup pyglacier
===============
...pyglacier version:  0.0.0.dev-15faa90
...fortran code directory:  ../glaciercode//
...fortran code version:  73b7f72
...default executable:  $CODEDIR/main.exe
...default glacier lib:  $CODEDIR/.obj/wrapper.so
...default namelist:  $CODEDIR/params.nml
...default output directory:  ./out/

You can then define a glacier class from restart file 
(or whatever netCDF file with the expected glacier fields):

    gl = pyglacier.Glacier.read('path/to/restart.nc')
    stress = gl.compute_stress()  # return a dimarray.Dataset of stress calculations
    gl.update_velocity()  # update velocity so that stress residual is zero
    stress2 = gl.compute_stress() # check the new stress balance

The glacier can also be defined more simply with four fields:

    gl = pyglacier.Glacier(x, zb, w, h, u)

and parameters can be changed:

    gl.set_param(name, value, group=group) # group is only needed if param name is ambiguous

A more exhaustive documentation will be made later. For now just check inline help for more information 
about glacier methods:

    help(pyglacier.Glacier)
