""" Check the mesh
"""
import numpy as np
import matplotlib.pyplot as plt
from wrapper import _fortran as f, _cwrap as cwrap

def check_func(func):

    x_gl = 100e3
    dx1 = 1000
    dx2 = 100
    L = 200e3
    D = 30e3
    N = 2000

    x_gl = 55e3
    dx1 = 1000.
    dx2 = 100.
    L = 2*x_gl
    D = 50e3
    N = int((x_gl-D)/dx1 + 2*D/dx2 + (L-x_gl-D)/dx1)

    mesh = np.empty(N, dtype="double")
    #f.mesh_any(c_double(x_gl), c_double(D), c_double(dx1), c_double(dx2), \
    #                 c_double(L), c_int(N), mesh.ctypes.data_as(POINTER(c_double)), my_c_char(func))
    cwrap(f.mesh_any)(x_gl, D, dx1, dx2, L, N, mesh, func)
    mesh = mesh[mesh<L]
    # plt.plot(mesh, marker='.')
    plt.plot(mesh[:-1], np.diff(mesh), marker='.', label="N={}".format(mesh.size))
    # plt.plot(mesh[:-2], np.diff(mesh[:-1])/np.diff(mesh[1:]), marker='.', label="N={}".format(mesh.size))
    plt.grid()
    plt.legend(frameon=False, loc="upper center")

    # for func in ["sigmoid", "arctan", "hyperbolic", "gaussian", "linear"]:
    # for func in ["gaussian"]:
    # for func in ["gaussian","sigmoid", "hyperbolic"]:
    #     plt.figure()
    #     check_func(func)
    #     plt.title(func)
    #     plt.savefig("work/"+func+'.pdf')
    # plt.show()
    # check_profile()
    # f.restype = int
    # f.glacier_init()
