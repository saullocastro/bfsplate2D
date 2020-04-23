import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigsh, spsolve
import numpy as np
from composites.laminate import read_stack

from bfsplate2d import (BFSPlate2D, update_KC0, update_KG, DOF, KC0_SPARSE_SIZE,
        KG_SPARSE_SIZE, DOUBLE, INT)
from bfsplate2d.quadrature import get_points_weights


def test_linear_buckling(plot_static=False, plot_lb=False):
    # number of nodes
    nx = 7 # along x
    ny = 5 # along y

    # getting integration points
    points, weights = get_points_weights(nint=4)

    # geometry
    a = 7 # along x
    b = 3 # along y

    # material properties
    E11 = 142.5e9
    E22 = 8.7e9
    nu12 = 0.28
    G12 = 5.1e9
    G13 = 5.1e9
    G23 = 5.1e9
    laminaprop = (E11, E22, nu12, G12, G13, G23)
    stack = [0, 45, -45, 90, 90, -45, 45, 0]
    ply_thickness = 0.001
    lam = read_stack(stack=stack, plyt=ply_thickness, laminaprop=laminaprop)

    # creating mesh
    x = np.linspace(0, a, nx)
    y = np.linspace(0, b, ny)
    xmesh, ymesh = np.meshgrid(x, y)

    # node coordinates and position in the global matrix
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T
    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    # identifying nodal connectivity for plate elements
    # similar than Nastran's CQUAD4
    #
    #   ^ y
    #   |
    #
    #  4 ________ 3
    #   |       |
    #   |       |   --> x
    #   |       |
    #   |_______|
    #  1         2


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
    KGr = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    init_k_KC0 = 0
    init_k_KG = 0

    plates = []
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        plate = BFSPlate2D()
        plate.c1 = DOF*nid_pos[n1]
        plate.c2 = DOF*nid_pos[n2]
        plate.c3 = DOF*nid_pos[n3]
        plate.c4 = DOF*nid_pos[n4]
        plate.ABD = lam.ABD
        plate.lex = a/(nx - 1)
        plate.ley = b/(ny - 1)
        plate.init_k_KC0 = init_k_KC0
        plate.init_k_KG = init_k_KG
        update_KC0(plate, points, weights, Kr, Kc, Kv)
        init_k_KC0 += KC0_SPARSE_SIZE
        init_k_KG += KG_SPARSE_SIZE
        plates.append(plate)

    KC0 = coo_matrix((Kv, (Kr, Kc)), shape=(N, N)).tocsc()

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

    # defining external force vector
    fext = np.zeros(KC0.shape[0], dtype=float)

    # applying unitary load along u at x=a
    # nodes at vertices get 1/2 the force
    Nxx = -1
    ftotal = Nxx*b # ny -> number of nodes along y
    check = (np.isclose(x, a) & ~np.isclose(y, 0) & ~np.isclose(y, b))
    fext[0::DOF][check] = ftotal/(ny - 1)
    check = ((np.isclose(x, a) & np.isclose(y, 0))
            |(np.isclose(x, a) & np.isclose(y, b)))
    fext[0::DOF][check] = ftotal/(ny - 1)/2

    Kuu = KC0[bu, :][:, bu]
    fextu = fext[bu]

    # static solver
    uu = spsolve(Kuu, fextu)
    u = np.zeros(KC0.shape[0], dtype=float)
    u[bu] = uu

    print('u extremes', u.min(), u.max())
    if plot_static:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        uplot = u[0::DOF].reshape(nx, ny).T
        levels = np.linspace(uplot.min(), uplot.max(), 300)
        plt.contourf(xmesh, ymesh, uplot, levels=levels)
        plt.colorbar()
        plt.show()

    # eigenvalue solver

    for plate in plates:
        update_KG(u, plate, points, weights, KGr, KGc, KGv)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]

    # solving modified generalized eigenvalue problem
    # Original: (KC0 + lambda*KG)*v = 0
    # Modified: (-1/lambda)*KC0*v = KG*v  #NOTE here we find (-1/lambda)
    num_eigenvalues = 5
    eigvals, eigvecsu = eigsh(A=KGuu, k=num_eigenvalues, which='SM', M=Kuu,
            tol=1e-6, sigma=1., mode='cayley')
    eigvals = -1./eigvals
    eigvecs = np.zeros((KC0.shape[0], num_eigenvalues), dtype=float)
    eigvecs[bu, :] = eigvecsu

    if plot_lb:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        mode = 0
        wplot = eigvecs[2::DOF, mode].reshape(nx, ny).T
        levels = np.linspace(wplot.min(), wplot.max(), 300)
        plt.contourf(xmesh, ymesh, wplot, levels=levels)
        plt.colorbar()
        plt.show()

    print('eigvals', eigvals)
    assert np.allclose(eigvals,[ 9469.11822737, 11706.06566922, 12835.65352185, 18537.3497283, 26327.59284375], rtol=1e-5)


if __name__ == '__main__':
    test_linear_buckling(plot_static=True, plot_lb=True)
