from scipy.linalg import eigh
import numpy as np
from composites.laminate import read_stack

from bfsplate2d import BFSPlate2D, update_K, update_M, DOF
from bfsplate2d.quadrature import get_points_weights

def test_nat_freq(plot_mode=None):
    # number of nodes
    nx = 7 # along x
    ny = 7 # along y

    # geometry
    a = 0.6
    b = 0.2

    # material properties (Aluminum)
    E = 70e9
    nu = 0.3
    rho = 7.8e3
    h = 0.001
    lam = read_stack(stack=[0], plyt=h, laminaprop=[E, nu], rho=rho)

    # creating mesh
    x = np.linspace(0, a, nx)
    y = np.linspace(0, b, ny)
    xmesh, ymesh = np.meshgrid(x, y)

    # node coordinates and position in the global matrix
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T
    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    nids_mesh = nids.reshape(nx, ny)

    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

    K = np.zeros((DOF*nx*ny, DOF*nx*ny))
    M = np.zeros((DOF*nx*ny, DOF*nx*ny))
    plates = []
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        plate = BFSPlate2D()
        plate.n1 = n1
        plate.n2 = n2
        plate.n3 = n3
        plate.n4 = n4
        plate.ABD = lam.ABD
        plate.h = h
        plate.rho = lam.rho
        update_K(plate, nid_pos, ncoords, K)
        update_M(plate, nid_pos, M)
        plates.append(plate)

    # applying boundary conditions
    # simply supported

    # locating nodes
    bk = np.zeros(K.shape[0], dtype=bool) # constrained DOFs, can be used to prescribe displacements

    x = ncoords[:, 0]
    y = ncoords[:, 1]

    # constraining w at all edges
    check = (np.isclose(x, 0.) | np.isclose(x, a) | np.isclose(y, 0.) | np.isclose(y, b))
    bk[2::DOF] = check
    # constraining u at x = 0
    check = np.isclose(x, 0.)
    bk[0::DOF] = check
    # constraining v at y = 0
    check = np.isclose(y, 0.)
    bk[1::DOF] = check

    # unconstrained nodes
    bu = ~bk # logical_not

    Kuu = K[bu, :][:, bu]
    Muu = M[bu, :][:, bu]

    # solving generalized eigenvalue problem
    num_eigenvalues = 5
    print('eig solver begin')
    eigvals, eigvecsu = eigh(a=Kuu, b=Muu)
    print('eig solver end')
    eigvecs = np.zeros((K.shape[0], num_eigenvalues), dtype=float)
    eigvecs[bu, :] = eigvecsu[:, :num_eigenvalues]
    omegan = eigvals**0.5

    # theoretical reference
    m = 1
    n = 1
    rho = lam.rho
    D = 2*h**3*E/(3*(1 - nu**2))
    wmn = (m**2/a**2 + n**2/b**2)*np.sqrt(D*np.pi**4/(2*rho*h))/2
    print('Theoretical omega123', wmn)
    print(omegan[:num_eigenvalues])
    assert np.isclose(omegan[0], wmn, rtol=0.01)

    if plot_mode is not None:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        wplot = eigvecs[2::DOF, plot_mode].reshape(nx, ny).T
        levels = np.linspace(wplot.min(), wplot.max(), 300)
        plt.contourf(xmesh, ymesh, wplot, levels=levels)
        plt.colorbar()
        plt.show()

