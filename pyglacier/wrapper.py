"""Wrapper around fortran's glacier code using ctypes
"""
from __future__ import division, print_function
import os, tempfile, warnings
from ctypes import CDLL, POINTER, c_int, c_double, c_char_p, c_bool, byref
# from numpy import empty, diff
import numpy as np

from .settings import GLACIERLIB  # first import
# from .oldnamelist import Namelist
from .namelist import Namelist

# Load glacier library
_fortran = None

def load_lib(lib):
    """ It is fine to load and switch across various libraries, but they
    must be physically in different places. Will fail after recompilation.
    """
    global _fortran
    _fortran = CDLL(lib)

# can be changed later
try:
    load_lib(GLACIERLIB)
except OSError as error:
    warnings.warn("Failed to load shared library. They may also be compilation problems")
    warnings.warn("Please setup project before loading pyglacier.setup(codedir...) or import from code dir")
    raise
    # warnings.warn("please reload lib with pyglacier.wrapper.load_lib('/path/to/wrapper.so') or restart project from within code directory")

# ======================================
# Functions to make ctypes easier to use
# ======================================

def _wraparray(a):
    return a.ctypes.data_as(POINTER(c_double))

def _cwrap(f):
    """ Wrapper to use with ctypes
    """
    # clen = 10  # assume length 10 for characters
    def func(*args):
        cargs = []
        for a in args:
            if isinstance(a, np.ndarray):
                # cargs.append(a.ctypes.data_as(POINTER(c_double)))
                cargs.append(_wraparray(a))
            elif type(a) is int:
                cargs.append(c_int(a))
            elif type(a) is float:
                cargs.append(c_double(a))
            elif type(a) is bool:
                cargs.append(c_bool(a))
            elif isinstance(a, basestring):
                # cargs.append(c_char_p(a[:clen]+" "*(clen-len(s))))
                cargs.append(c_char_p(a))
            else:
                print("warning: unknown ctype equivalent for "+type(a))
                cargs.append(a)
            # print "Convert: ",a,"==>",cargs[-1]
        # print "Input arguments",args
        # print "C-wrapped arguments:",cargs
        return f(*cargs)
        #return cargs
    return func

# ============================================================================
# Functional functions that do not need any glacier instance being initialized
# ============================================================================

def smooth(x1, m):
    """ Smooth an array with an exponential convolution
    
    Parameters
    ----------
    x1 : numpy array 1d
    m : int
        half-width of the filter (corresponding to 3-sigma)
    
    Returns
    -------
    x2 : smoothed array
    """
    x2 = np.empty_like(x1, dtype="double")
    _cwrap(_fortran.smooth_array)(x1.size, x1, x2, m)
    return x2

# Compute new elevation and grounding line position
# -------------------------------------------------
def apply_archimede_func(x, H, zb, rho_sw=1000., rho_i=910.):
    """Apply archimede on the glacier...

    Parameters
    ----------
    x : x axis
    H : thickness
    zb : bedrock elevation
    rho_sw, rho_i : float, optional
        seawater and ice densities

    Returns
    -------
    hb : bottom elevation
    hs : surface elevation
    gl : 0-based index of last cell preceding the grounding line
    xgl : interpolated grounding line position (between cells)
    """
    hb = np.empty_like(H)
    hs = np.empty_like(H)
    gl = c_int()
    xgl = c_double()
    _fortran.apply_archimede_func(_wraparray(x), _wraparray(H), _wraparray(zb), 
                                   c_double(rho_sw), c_double(rho_i),
                                   _wraparray(hs), _wraparray(hb), byref(gl), byref(xgl), c_int(x.size))
    return hb, hs, gl.value-1, xgl.value

# def compute_stress_func(c, x, W, zb, H, beta, A, n, m, g, rho_i, rho_sw, 
#                           bc_upstream_type=1, bc_upstream_value=0., U0=None, 
#                           stress_only=False):
def compute_velocity_func(x, W, zb, H, A, n, beta, m, g, rho_i, rho_sw, 
                          bc_upstream_type=1, bc_upstream_value=0., U0=None, 
                          stress_only=False):
    """ Compute velocity

    Parameters
    ----------
    x : x position
    W : glacier width
    zb : bedrock elevation
    H : glacier thickness
    A : scalar or array-like
        rate factor (to compute viscosity)
    n : int
        flow law exponent
    beta : scalar or array-like
        basal drag parameter ( U = beta*U**|1/m-1|*U = beta2 U)
    m : basal friction exponent
    g : gravity constant
    rho_i : ice density
    rho_sw : sea water density
    bc_upstream_type : 1 (velocity U[0]) or 2 (gradient on U dU/dx[0])
    bc_upstream_value : U[0] or dU/dx[0]
    U0 : scalar or array-like, optional
        initial condition for U
    stress_only : bool, optional
        if True, do not compute velocity but only compute the stress
        (only make sense if U is provided)

    Returns
    -------
    U : new velocity 
    , stress_driving, stress_long, stress_lat, stress_basal, stress_residual
    """
    if U0 is None:
        U = np.empty.like(x)
    elif np.isscalar(U0):
        U = np.empty_like(x)
        U.fill(U0)
    else:
        U = U0.copy()

    if np.isscalar(A):
        A0 = A
        A = np.empty_like(x)
        A.fill(A0)

    if np.isscalar(beta):
        beta0 = beta
        beta = np.empty_like(x)
        beta.fill(beta0)

    stress_driving = np.empty_like(x)
    stress_long = np.empty_like(x)
    stress_lat = np.empty_like(x)
    stress_basal = np.empty_like(x)
    stress_residual = np.empty_like(x)

    _fortran.compute_velocity_func(c_int(x.size), _wraparray(x), _wraparray(W), _wraparray(zb), _wraparray(H), 
                                   _wraparray(beta), _wraparray(A), c_int(n), c_int(m), 
                                   c_double(g), c_double(rho_i), c_double(rho_sw), 
                                   _wraparray(U), 
                                   _wraparray(stress_driving), _wraparray(stress_long), _wraparray(stress_lat), 
                                   _wraparray(stress_basal), _wraparray(residual), c_bool(stress_only))

    return U, stress_driving, stress_long, stress_lat, stress_basal, stress_residual

# ============================================
# Functions below act on a full glacier object
# that lives in fortran memory, within an array of 1000 
# glaciers that can co-exist simultaneously.
# ============================================

# Switch glacier which is acted upon
# ==================================

associate_glacier = _cwrap(_fortran.associate_glacier)

# Parameters
# ==========

# I/O between fortran model and disk
# ----------------------------------
def read_params(fname):
    _cwrap(_fortran.read_params)(fname, len(fname))
def write_params(fname):
    _cwrap(_fortran.write_params)(fname, len(fname))
def read_control(fname):
    _cwrap(_fortran.read_control)(fname, len(fname))
def write_control(fname):
    _cwrap(_fortran.write_control)(fname, len(fname))

# I/O between _fortran model and python
# -------------------------------------
def set_nml(nml):
    """ Put namelist to fortran model
    """
    id_, fn = tempfile.mkstemp() # tempfile
    nml.write(fn)
    read_params(fn)
    os.remove(fn) # remove temp file

def get_nml():
    """ Get namelist from fortran model
    """
    id_, fn = tempfile.mkstemp() # tempfile
    write_params(fn)
    nml = Namelist.read(fn)
    os.remove(fn) # remove temp file
    return nml

def get_param(group, name):
    nml = get_nml()
    return nml.filter(group=group, name=name)[0].value

def set_param(group, name, value):
    nml = get_nml()  # read existing parameters
    nml.filter(group=group, name=name)[0].value = value
    set_nml(nml)

# Type specific functions (no need for namelist I/O)
# -------------------------------------------------
def set_param_control(name, value):
    value = str(value)
    _cwrap(_fortran.set_param_control)(name, len(name), value, len(value))

def set_param_geometry(name, value):
    value = str(value)
    _cwrap(_fortran.set_param_geometry)(name, len(name), value, len(value))

def set_param_dynamics(name, value):
    value = str(value)
    _cwrap(_fortran.set_param_dynamics)(name, len(name), value, len(value))

def set_param_smb(name, value):
    value = str(value)
    _cwrap(_fortran.set_param_smb)(name, len(name), value, len(value))

def set_param_fjormelt(name, value):
    value = str(value)
    _cwrap(_fortran.set_param_fjormelt)(name, len(name), value, len(value))

def set_param_basalmelt(name, value):
    value = str(value)
    _cwrap(_fortran.set_param_basalmelt)(name, len(name), value, len(value))

def set_param_calving(name, value):
    value = str(value)
    _cwrap(_fortran.set_param_calving)(name, len(name), value, len(value))

def update_params(params):
    """ params: dict of dict
    """
    nml = get_nml()  # read existing parameters
    for g in params:
        G = g.upper()
        if G not in nml.groups.keys():
            raise ValueError("{} group does not exist".format(G))
        for k in params[g]:
            K = k.upper()
            if K not in nml.groups[G].keys():
                raise ValueError("{} param does not exist in {}".format(K, G))
            nml.groups[G][K] = params[g][k]
    set_nml(nml)

# control variables (years etc...) are not in the namelist
def get_nml_control():
    id_, fn = tempfile.mkstemp() # tempfile
    write_control(fn)
    nml = Namelist.read(fn)
    os.remove(fn) # remove temp file
    return nml

# def get_param_control(self, name):
#     _cwrap(_fortran.get_param_control, name, len(name), value)
#     return value


# variables
# =========

def get_size():
    return _fortran.get_size()

# I/O between fortran model and disk
# ----------------------------------
def read_state(fname):
    _cwrap(_fortran.read_state)(fname, len(fname))

def write_state(fname):
    _cwrap(_fortran.write_state)(fname, len(fname))

# Exchange between fortran model and python
# -----------------------------------------
def get_var(name, dtype="double", glacierid=None):
    n = _fortran.get_size()
    x = np.empty(n, dtype=dtype)
    getattr(_fortran,"get_"+name.lower())(_wraparray(x), c_int(x.size))
    return x

def set_var(name, x):
    x = np.asarray(x)
    n = _fortran.get_size()
    if x.size != n:
        print( "Expected:", n)
        print( "Got:", x.size)
        raise ValueError("Sizes do not match")
    getattr(_fortran,"set_"+name.lower())(_wraparray(x), c_int(x.size))

def get_gl(): 
    return _fortran.get_gl()

def set_gl(gl): 
    return _fortran.set_gl(c_int(gl))

def set_c(c): 
    _fortran.set_c(c_int(c))

def get_c(): 
    return _fortran.get_c()

_fortran.get_xgl.restype = c_double
_fortran.get_xc.restype = c_double
def get_xgl():
    return _fortran.get_xgl()
def get_xc():
    return _fortran.get_xc()

def set_xc(x):
    _fortran.set_xc(c_double(x))
def set_xgl(x):
    _fortran.set_xgl(c_double(x))


# Initialize, integrate
# =====================

def allocate(n):
    _fortran.glacier_allocate(c_int(n))

def init(params_nml=None, input_nc=None):
    """
    optional:
        params_nml: alternative namelist file
        input_nc: initialize the glacier from that file (will set geo%mode to "from_file")
            (and will define as reference geometry)
    """
    _cwrap(_fortran.init)(params_nml, len(params_nml), input_nc, len(input_nc))

def integrate(n, dt=3.65, out_dir="", out_freq=None, out_mult=None, rst_freq=None, rst_mult=None):
    """ 
    n : int
        number of iterations
    dt : float
        timestep in days (default to 3.65 j)
    """
    out_dir = out_dir or ""  # if empty, will not print
    if out_freq is not None: set_param_control("out_freq", out_freq)
    if out_mult is not None: set_param_control("out_mult", out_mult)
    if rst_freq is not None: set_param_control("rst_freq", rst_freq)
    if rst_mult is not None: set_param_control("rst_mult", rst_mult)
    if out_dir and not os.path.exists(out_dir):
        print("Create output directory: "+out_dir)
        os.makedirs(out_dir)
    _fortran.integrate(c_int(n), c_double(dt*24*3600.), out_dir, len(out_dir))

def interp(x):
    """ interpolate glacier on new grid
    """
    _cwrap(_fortran.glacier_interp)(x, x.size)
  # subroutine f_tune_beta(beta_min, beta_max, w)  bind(c, name="tune_beta")

# functions without arguments
refresh = _fortran.refresh
update_velocity = _fortran.update_velocity
update_massbalance = _fortran.update_massbalance
update_flotation = _fortran.update_flotation
glacier_calving = _fortran.glacier_calving
compute_stress = _fortran.compute_stress
glacier_interp_gl = _fortran.glacier_interp_gl

#==============================================
# Make it easier to access field as array-like
#==============================================
_VARIABLES = ["h","u","hs","hb","zb","w", "dx"]
_STRESS_VARIABLES = ["driving", "lat", "long", "basal", "residual"]
_MB_VARIABLES = ["smb","basalmelt","fjordmelt", "dynmb"]  # mass balance
_ALL_VARIABLES = _VARIABLES + [s + '_stress' for s in _STRESS_VARIABLES] + _MB_VARIABLES

class _FGlacierVar(object):
    def __init__(self, name):
        self.name = name
    def __getitem__(self, idx):
        return get_var(self.name)[idx]
    def __setitem__(self, idx, val):
        assert idx is None or idx == slice(None), 'can only set full arrays'
        return set_var(self.name, val)
    @property
    def size(self):
        return _fortran.get_size()
    @property
    def shape(self):
        return (self.size,)
    ndim = 1
    def __array__(self):
        return self[:]

for v in ['x'] + _ALL_VARIABLES:
    globals()[v] = _FGlacierVar(v)

    # also some aliases (e.g. remove stress suffix)
    if v.endswith('_stress'):
        globals()[v[:7]] = globals()[v]

def test():
    # some testing
    import matplotlib.pyplot as plt
    # gl = _FGlacier()
    init()
    integrate(300, dt=3.65)
    gl.plot()
    plt.show()

if __name__ == "__main__":
    # TO EXECUTE: python -m pyglacier.wrapper
    test()
