import sys
sys.path.append('..')

from scipy.sparse import coo_matrix
from scipy.linalg import eigh
import numpy as np
from composites.laminate import read_stack

from bfsplate2d import (BFSPlate2D, update_KC0, update_M, DOF, KC0_SPARSE_SIZE,
        M_SPARSE_SIZE, DOUBLE, INT)
from bfsplate2d.quadrature import get_points_weights

def test_nat_freq(plot_mode=None):
    # number of nodes
    nx = 9 # along x
    ny = 7 # along y

    points, weights = get_points_weights(nint=4)

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

    num_elements = len(n1s)
    print('num_elements', num_elements)

    N = DOF*nx*ny
    Kr = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kc = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kv = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    Mr = np.zeros(M_SPARSE_SIZE*num_elements, dtype=INT)
    Mc = np.zeros(M_SPARSE_SIZE*num_elements, dtype=INT)
    Mv = np.zeros(M_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    init_k_KC0 = 0
    init_k_M = 0

    plates = []
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        plate = BFSPlate2D()
        plate.c1 = DOF*nid_pos[n1]
        plate.c2 = DOF*nid_pos[n2]
        plate.c3 = DOF*nid_pos[n3]
        plate.c4 = DOF*nid_pos[n4]
        plate.ABD = lam.ABD
        plate.h = h
        plate.rho = lam.rho
        plate.lex = a/(nx - 1)
        plate.ley = b/(ny - 1)
        plate.init_k_KC0 = init_k_KC0
        plate.init_k_M = init_k_M
        update_KC0(plate, points, weights, Kr, Kc, Kv)
        update_M(plate, Mr, Mc, Mv)
        init_k_KC0 += KC0_SPARSE_SIZE
        init_k_M += M_SPARSE_SIZE
        plates.append(plate)

    KC0 = coo_matrix((Kv, (Kr, Kc)), shape=(N, N)).tocsc()
    M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()

    # applying boundary conditions
    # simply supported

    # locating nodes
    bk = np.zeros(KC0.shape[0], dtype=bool) # constrained DOFs, can be used to prescribe displacements

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

    Kuu = KC0[bu, :][:, bu]
    Muu = M[bu, :][:, bu]

    # solving generalized eigenvalue problem
    num_eigenvalues = 4
    print('eig solver begin')
    eigvals, eigvecsu = eigh(a=Kuu.toarray(), b=Muu.toarray(), eigvals=(0,
        num_eigenvalues-1))
    print('eig solver end')
    eigvecs = np.zeros((KC0.shape[0], num_eigenvalues), dtype=float)
    eigvecs[bu, :] = eigvecsu
    omegan = eigvals**0.5

    # theoretical reference
    m = 1
    n = 1
    rho = lam.rho
    D = 2*h**3*E/(3*(1 - nu**2))
    wmn = (m**2/a**2 + n**2/b**2)*np.sqrt(D*np.pi**4/(2*rho*h))/2
    print('Theoretical omega123', wmn)
    print(omegan)
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

