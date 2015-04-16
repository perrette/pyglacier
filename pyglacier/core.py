""" Make a glacier class which uses fortran's code

The main difference with pyglacier.wrapper's Glacier class is that the latter
does not store state information (instead, all is in fortran), whereas
the Glacier class of this module is a true class, which can be duplicated and
so on, and all necessary state information and parameters are passed to the 
fortran code every time a function (e.g. update_velocity, update_mass...) 
is called.
"""
from __future__ import division, print_function
import os, tempfile, datetime, shutil, copy
import warnings

import numpy as np
import dimarray as da

# useful for some in-memory operations
import pyglacier
try:
    from . import wrapper
except Exception as error:
    warnings.warn("Could not import wrapper. Please set the paths correctly or check the code version.")
    wrapper = None

# from .settings import OUTDIR, NML
from . import settings
from .namelist import Namelist, Param
# from .namelist import Namelist, read_namelist_file
from .plotting import plot_glacier
from .run import run_model
# from .wrapper import fortran, get_var, set_var, set_nml, get_nml
from .plotting import plot_glacier, plot_stress

def load_default_nml(params_nml):
    nml = Namelist.read(params_nml) # default
    # make sure geometry is always taken from file 
    # and not from some default profile
    i = nml.index(Param('mode', group='geometry'))
    nml[i].value = 'from_file'  
    globals()['DEFAULT_PARAMS'] = nml # make it module-wide
    return nml

try:
    load_default_nml(settings.NML)
except Exception as error:
    warnings.warn("Could not load default namelist")
    warnings.warn("Please setup project before loading pyglacier.setup(codedir...) or import from code dir")
    raise

def _get_param(params, name, group=None):
    " get one Param instance from the list of params"
    if group is not None:
        i = params.index(Param(name, group=group))
        param = params[i]
    else:
        matches = [p for p in params if p.name == name]
        assert len(matches) == 1, 'no or multiple params found, please provide group: \n'+repr(matches)
        param = matches[0]
    return param

class Glacier(object):

    _default_params = DEFAULT_PARAMS
    _class_directories = []  # history of output directories for simulation for all instances

    def __init__(self, x, zb, w, h, u, params=None, beta=None, id=None):
        self.x = np.asarray(x)
        self.zb = np.asarray(zb)
        self.w = np.asarray(w)
        self.h = np.asarray(h)
        self.u = np.asarray(u)
        self._beta = np.asarray(beta) if beta is not None else None
        assert self.x.shape == self.zb.shape == self.w.shape == self.h.shape == self.u.shape, 'inconsistent shapes !'
        self._params = params # see params property

        self.outdir = None
        self._instance_directories = [] # keep track of what was created by this instance
        self.id = id # for in-memory calculations

    @property
    def size(self):
        return self.x.size

    @property
    def dx(self):
        return np.gradient(self.x)

    # Convenience methods to set / get param values 
    # and to common read / write / update operations
    # =============================================
    def set_param(self, name, value, group=None):
        " set param values "
        param = _get_param(self.params, name, group)
        param.value = value

    def get_param(self, name, group=None):
        param = _get_param(self.params, name, group)
        return param.value

    # Here I/O of whole bunch of params
    def update_params(self, params):
        for p in Namelist(params):
            i = self.params.index(p)
            self.params[i] = p

    def read_params(self, nml):
        params = Namelist.read(nml)
        self.update_params(params)

    def write_params(self, nml):
        self.params.write(nml)

    # make params a property to check concistency with beta
    @property
    def params(self):
        # user-defined basal friction?
        params = self._params or Namelist(self._default_params)
        if self._beta is not None:
            params.search(name='basal_mode').value = 'prescribed'
        else:
            params.search(name='basal_mode').value = 'constant'
        return params

    @params.setter
    def params(self, params):
        self._params = params

    # Convert to / from dataset
    # =============================================
    _names = 'zb', 'W', 'H', 'U'  # model output: small and big letters...
    def to_dataset(self, compute_elevation=False):
        ds = da.Dataset()
        for v in self._names:
            ds[v] = getattr(self, v.lower())
            # ds[v] = da.DimArray(getattr(self, v.lower()), axes=[self.x], dims=['x'])
        # also add glacier elevation gl, xgl
        if compute_elevation:
            hb, hs, gl, xgl = self.compute_elevation()
            ds['hb'] = hb
            ds['hs'] = hs
            ds['gl'] = gl
            ds['xgl'] = xgl

        # check basal mode
        if self._beta is not None:
            ds['beta'] = self._beta
        
        ds.set_axis(self.x, name='x', inplace=True)
        return ds

    @classmethod
    def from_dataset(cls, ds):
        return cls(ds.x, ds['zb'], ds['W'], ds['H'], ds['U'], beta=ds['beta'] if 'beta' in ds else None)

    # Convenience methods
    # =========================================
    @classmethod
    def read(cls, fname):
        " read from restart file "
        ds = da.read_nc(fname)
        return cls.from_dataset(ds)

    @classmethod
    def read_output(cls, fname, t=-1):
        ds = da.read_nc(fname, cls._names, indices={'time':t}, indexing='position')
        return cls.from_dataset(ds)


    # Run the model on disk
    # =============================================
    def _get_out_dir(self, create=False):
        """Generate a directory under default output directory
        """
        outdir = os.path.join(settings.OUTDIR, str(datetime.datetime.now())).replace(" ",".").replace(":","")
        if not os.path.exists(outdir) and create:
            os.makedirs(outdir)
        return outdir

    def _get_in_file(self):
        """ reference geometry, forcing and so on
        """
        return os.path.join(self.out_dir, "glacier_input.nc")

    def integrate(self, years, out_dir=None, **kwargs):
        """Run the model as it were from command-line

        years : float
            years of simulation
        out_dir : str, optional
            provide an output directory for the simulation
            (otherwise a new directory will be created)
        **kwargs : passed to run_model
            This includes 'dt' (time step etc...)
        """
        if out_dir:
            self.out_dir = out_dir
        else:
            self.out_dir = self._get_out_dir(create=True) # new out directory
            self._instance_directories.append(self.out_dir)
            self._class_directories.append(self.out_dir)

        # write glacier state to disk
        in_file = self._get_in_file()
        self.to_dataset().write_nc(in_file)

        # and use it both as a restart and file and reference geometry file
        res = run_model(years, params=self.params, out_dir=self.out_dir, in_file=in_file, rst_file=in_file, **kwargs)
        if res != 0:
            warnings.warn("result value is not 0. Problem during integration?")
            return None

        # read results from restart file
        ds = da.read_nc(os.path.join(self.out_dir, 'restart.nc'))
        gl = self.from_dataset(ds)

        # read results from output file, whose geometry extends further
        # ds = self.read_output(os.path.join(self.out_dir, 'output.nc'))
        gl.params = self.params  # copy params
        return gl

    def regrid(self, x):
        ds = self.to_dataset().interp_axis(x)
        gl = self.from_dataset(ds)
        gl.params = self.params
        return gl

    # Methods to remove created directories
    # =================================================
    def clean_outdir(self):
        shutil.rmtree(self.outdir)

    def clean_directories_class(self, indices=None, force=False):
        """ e.g. 
        glacier.clean_directories_class([-1]) # remove last simulation
        glacier.clean_directories_class(range(4)) # remove first 4
        glacier.clean_directories_class(force=True)  # remove all silently
        """
        if indices is None:
            indices = range(len(cls._class_directories))
        directories = [cls._class_directories[i] for i in indices]
        return self.__clean_directories(self._class_directories, indices, force)

    def clean_directories_instance(self, force=False):
        """ Remove all directories which this glacier created
        """
        return self._clean_directories(self._instance_directories, force)

    def _clean_directories(self, directories, force):
        """
        indices : index for created_directories
        force : do not ask user before deleting directories
        """
        for d in directories:
            if not os.path.exists(d):
                print(d,"directory does not exist, skip")
        directories = [d for d in directories if os.path.exists(d)]

        if len(directories) == 0:
            print("Nothing to clean")
            return
        if not force:
            print("About to remove simulation directories:",directories)
            ans = raw_input('OK? ( y / [n] )')
            if ans.lower() != 'y':
                print("Cancelled by user.")
                return 
        for d in directories:
            print("rm "+d)
            shutil.rmtree(d)

        self._autoclean_created_directories()
        return

    def _autoclean_created_directories(self):
        cls = self.__class__
        cls._class_directories = [d for d in cls._class_directories if os.path.exists(d)]
        self._instance_directories = [d for d in self._instance_directories if os.path.exists(d)]

    # Handy in-memory operations
    # =========================================

    def _in_memory_init(self, id=None, fixedgrid=True):
        """ Set glacier parameters and state variables
        """
        if id is not None:
            wrapper.associate_glacier(id)
            self.id = id  
        dir_name = tempfile.mkdtemp(prefix='glacier_init') # tempfile
        params_nml = os.path.join(dir_name, 'params.nml')
        input_nc = os.path.join(dir_name, 'input.nc')

        if fixedgrid:
            pars = Namelist([p for p in self.params])
            pars.search(name="interp_mode").value="none"
            pars.write(params_nml) # write params to file
        else:
            self.params.write(params_nml) # write params to file
        self.to_dataset().write_nc(input_nc)
        wrapper.init(params_nml, input_nc)
        shutil.rmtree(dir_name)

    @classmethod
    def from_memory(self, id=None):
        """ retrieve glacier state variable from fortran code
        """
        if id is not None:
            wrapper.associate_glacier(id)

        ds = da.Dataset()
        for v in self._names:
            ds[v] = wrapper.get_var(v)
        ds.set_axis(wrapper.get_var('x'), name='x', inplace=True)
        gl = self.from_dataset(ds)
        gl.params = wrapper.get_nml()
        return gl

    def update_velocity(self, init=True):
        """
        init : True by default
            reinitialize in-memory glacier (read params, netcdf etc...)
        """
        interp_mode = self.get_param("interp_mode")
        self.set_param("interp_mode","none")
        if init:
            self._in_memory_init(self.id) # set a in-memory glacier that is ready for further computation
        wrapper.update_velocity()
        self.u = wrapper.get_var('u') # do not include last value...
        self.set_param("interp_mode",interp_mode) # back to initial value

    # def compute_velocity(self):
    #     wrapper.compute_velocity_func(x, W, zb, H, A, n, beta, m, g, rho_i, rho_sw, 
    #                       bc_upstream_type=1, bc_upstream_value=0., U0=None, 
    #                       stress_only=False):
    #
    def compute_elevation(self):
        rho_sw = self.get_param('rho_sw')
        rho_i = self.get_param('rho_i')
        hb, hs, gl, xgl = wrapper.apply_archimede_func(self.x, self.h, self.zb, rho_sw, rho_i)
        return hb, hs, gl, xgl

    # quick access to gl, xgl and so on
    # =================================
    @property
    def gl(self):
        hb, hs, gl, xgl = self.compute_elevation()
        return gl

    @property
    def xgl(self):
        hb, hs, gl, xgl = self.compute_elevation()
        return xgl

    @property
    def hs(self):
        hb, hs, gl, xgl = self.compute_elevation()
        return hs

    @property
    def hb(self):
        hb, hs, gl, xgl = self.compute_elevation()
        return hb

    def calve(self):
        """ check calving front position
        """
        self._in_memory_init(self.id) # set a in-memory glacier that is ready for further computation
        wrapper.glacier_calving()
        # xc = wrapper.get_xc()
        self.h = wrapper.get_var('h') # update H
        # return wrapper.get_xc()

    #
    # Calving front: should be representative of 
    #
    @property
    def c(self):
        self._in_memory_init(self.id) # set a in-memory glacier that is ready for further computation
        return wrapper.get_c()

    @property
    def xc(self):
        return wrapper.get_var('x')[wrapper.get_c()] # will do it for now

    def compute_stress(self, init=True):
        """ compute stress associated with current velocity and glacier profile
        """
        if init:
            self._in_memory_init(self.id) # set a in-memory glacier that is ready for further computation
        wrapper.compute_stress()
        ds = da.Dataset()
        _stress_variables = ["driving", "lat", "long", "basal", "residual"]
        for v in [s + '_stress' for s in _stress_variables]:
            ds[v] = wrapper.get_var(v)
        ds.set_axis(wrapper.get_var('x'), name='x', inplace=True)
        return ds

    # quick access to stress properties
    # =================================
    @property
    def driving_stress(self):
        ds = self.compute_stress()
        return ds['driving_stress']
    @property
    def lat_stress(self):
        ds = self.compute_stress()
        return ds['lat_stress']
    @property
    def long_stress(self):
        ds = self.compute_stress()
        return ds['long_stress']
    @property
    def basal_stress(self):
        ds = self.compute_stress()
        return ds['basal_stress']
    @property
    def residual_stress(self):
        ds = self.compute_stress()
        return ds['residual_stress']

    def compute_mass_balance(self, init=True):
        if init:
            self._in_memory_init(self.id) # set a in-memory glacier that is ready for further computation
        wrapper.update_massbalance()  # compute all mass balance fluxes

        ds = da.Dataset()
        _variables = ["smb","basalmelt","fjordmelt", "dynmb"]  # mass balance
        for v in _variables:
            ds[v] = wrapper.get_var(v)
        ds.set_axis(wrapper.get_var('x'), name='x', inplace=True)
        return ds

    @property
    def smb(self):
        ds = self.compute_mass_balance()
        return ds['smb']

    @property
    def basalmelt(self):
        ds = self.compute_mass_balance()
        return ds['basalmelt']

    @property
    def fjordmelt(self):
        ds = self.compute_mass_balance()
        return ds['fjordmelt']

    @property
    def dynmb(self):
        ds = self.compute_mass_balance()
        return ds['dynmb']

    def integrate_in_memory(self, timesteps, dt=3.65, out_dir=None, 
                            out_freq="none", out_mult=1, rst_freq="none", rst_mult=1, 
                            init=True, fixedgrid=False):
        """ Similar to integrate, but using the wrapper so that no output
        has to be generated on disk. Note number of timesteps has to be indicated here.
        """
        if init:
            self._in_memory_init(self.id, fixedgrid=fixedgrid)

        # If out-dir is None, do not produce outputs !
        if out_dir is None:
            wrapper.set_param_control('out_freq', 'none')
            wrapper.set_param_control('rst_freq', 'none')
        wrapper.integrate(timesteps, dt=dt, out_dir=out_dir, out_freq=out_freq, out_mult=out_mult, rst_freq=rst_freq, rst_mult=rst_mult)

        # Now generate another glacier
        gl = self.from_memory(self.id)
        return gl

    def plot(self, **kwargs):
        return plot_glacier(self.to_dataset(compute_elevation=True), **kwargs)

    def plot_stress(self, **kwargs):
        return plot_stress(self.compute_stress(), **kwargs)

    def plot_massbalance(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        scal = 365*3600*24
        ds = self.compute_mass_balance()*scal
        res = ds['smb']+ds['dynmb']-ds['fjordmelt']-ds['basalmelt']
        ax.plot(self.x, ds['dynmb'], label='dynmb', **kwargs)
        ax.plot(self.x, ds['smb'], label='smb', **kwargs)
        ax.plot(self.x, ds['fjordmelt'], label='fjordmelt', **kwargs)
        ax.plot(self.x, res, label='residual', **kwargs)
        ax.set_ylabel("meters / year")

# # Higher-level exchange that operates on datasets (builds on above functions)
# # --------------------------------------------------------------------------
# # variables defining the glacier state (to exchange with fortran code by default)
# _VARIABLES = ["h","u","hs","hb","zb","w", "dx"]
# _STRESS_VARIABLES = ["driving", "lat", "long", "basal", "residual"]
# _MB_VARIABLES = ["smb","basalmelt","fjordmelt", "dynmb"]  # mass balance
# _ALL_VARIABLES = _VARIABLES + [s + '_stress' for s in _STRESS_VARIABLES] + _MB_VARIABLES
#
# def get_summary():
#     """ Get all possible variables from the fortran code, as a single dataset
#     """
#     ds = da.Dataset()
#     x = get_var('x')
#     for nm in _ALL_VARIABLES:
#         v = get_var(nm)
#         ds[nm] = da.DimArray(v, axes=[x], dims=['x'])
#
#     ds['c'] = get_c()
#     ds['gl'] = get_gl()
#     ds['xc'] = get_xc()
#     ds['xgl'] = get_xgl()
#     return ds

# class FGlacier(Glacier):
#     """ Glacier that is bound to fortran directly via ctype
#     """
#     _MAXID = 1000 # hard-coded in the wrapper
#     current_id = 1
#     def __init__(self, x, zb, w, h, u, params=None, id=None):
#         Glacier.__init__(self, x, zb, w, h, u, params)
#         if id is not None:
#             assert id <= self._MAXID
#             self.__class__.current_id = id  # maxid
#         self.id = self.current_id
#
#     def integrate(self, glacierid=1):

# # # plot glacier state
# def plot_glacier():
#     ds = get_state()
#     return _plot_glacier(ds)
#
# def plot_stress():
#     ds = get_stress()
#     return _plot_stress(ds)
#
