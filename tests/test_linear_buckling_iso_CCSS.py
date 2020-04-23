import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, spsolve
import numpy as np
from composites.laminate import read_stack

from bfsplate2d import (BFSPlate2D, update_KC0, update_KG, DOF, KC0_SPARSE_SIZE,
        KG_SPARSE_SIZE, DOUBLE, INT)
from bfsplate2d.quadrature import get_points_weights


def test_linear_buckling_iso_CCSS(plot_static=False, plot_lb=False):
    """See Bruhn Fig. C5.2, case C with clamped loaded edges
    """
    # number of nodes
    nx = 5 # along x
    ny = 5 # along y

    # getting integration points
    nint = 4
    points, weights = get_points_weights(nint=nint)

    # geometry
    a = 3 # along x
    b = 3 # along y

    # material properties
    E = 200e9
    nu = 0.3
    laminaprop = (E, E, nu)
    stack = [0]
    h = 0.001
    lam = read_stack(stack=stack, plyt=h, laminaprop=laminaprop)

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
        plate.n1 = n1
        plate.n2 = n2
        plate.n3 = n3
        plate.n4 = n4
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

    # locating nodes
    bk = np.zeros(KC0.shape[0], dtype=bool) # constrained DOFs, can be used to prescribe displacements

    x = ncoords[:, 0]
    y = ncoords[:, 1]

    # applying boundary conditions
    # simply supported
    check = isclose(x, 0) | isclose(x, a) | isclose(y, 0) | isclose(y, b)
    bk[2::DOF] = check
    check = isclose(x, 0) | isclose(x, a)
    bk[3::DOF] = check
    # point supports
    check = isclose(x, a/2) & (isclose(y, 0) | isclose(y, b))
    bk[0::DOF] = check
    check = isclose(y, b/2) & (isclose(x, 0) | isclose(x, a))
    bk[1::DOF] = check

    # unconstrained nodes
    bu = ~bk # logical_not

    # defining external force vector
    fext = np.zeros(KC0.shape[0], dtype=float)

    # applying unitary load along u at x=a
    # nodes at vertices get 1/2 the force
    for plate in plates:
        pos1 = nid_pos[plate.n1]
        pos2 = nid_pos[plate.n2]
        pos3 = nid_pos[plate.n3]
        pos4 = nid_pos[plate.n4]
        if isclose(x[pos3], a):
            Nxx = -1
            xi = +1
        elif isclose(x[pos1], 0):
            Nxx = +1
            xi = -1
        else:
            continue
        lex = plate.lex
        ley = plate.ley
        indices = []
        c1 = DOF*pos1
        c2 = DOF*pos2
        c3 = DOF*pos3
        c4 = DOF*pos4
        cs = [c1, c2, c3, c4]
        for ci in cs:
            for i in range(DOF):
                indices.append(ci + i)
        fe = np.zeros(4*DOF, dtype=float)
        for j in range(nint):
            eta = points[j]
            plate.update_Nu(xi, eta)
            Nu = np.asarray(plate.Nu)
            fe += ley/2*weights[j]*Nu*Nxx
        fext[indices] += fe

    Kuu = KC0[bu, :][:, bu]
    fextu = fext[bu]

    # static solver
    uu = spsolve(Kuu, fextu)
    u = np.zeros(KC0.shape[0], dtype=float)
    u[bu] = uu

    if plot_static:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        uplot = u[0::DOF].reshape(nx, ny).T
        vplot = u[1::DOF].reshape(nx, ny).T
        print('u extremes', uplot.min(), uplot.max())
        print('v extremes', vplot.min(), vplot.max())
        levels = np.linspace(uplot.min(), uplot.max(), 300)
        plt.contourf(xmesh, ymesh, uplot, levels=levels)
        plt.colorbar()
        plt.show()

    # eigenvalue solver

    # getting integration points
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

    kc = eigvals[0]/(E*np.pi**2*(h/b)**2/(12*(1 - nu**2))*h)
    assert isclose(kc, 6.6, rtol=0.05)


if __name__ == '__main__':
    test_linear_buckling_iso_CCSS(plot_static=True, plot_lb=True)
