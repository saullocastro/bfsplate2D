import sys
sys.path.append(r'C:\repositories\bfsplate2d')

import numpy as np
from numpy import isclose
from scipy.linalg import eigh, solve
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from composites.laminate import read_stack

from bfsplate2d import BFSPlate2D, update_K, update_Kg
from bfsplate2d.quadrature import get_points_weights

DOF = 6
def test_linear_buckling_iso_SSSS(plot_static=False, plot_lb=False, nx=9, ny=5):
    """See Bruhn Fig. C5.2 all simply supported
    """
    # number of nodes
    nx = nx # along x
    ny = ny # along y

    # geometry
    a = 11 # along x
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

    Kg = np.zeros((DOF*nx*ny, DOF*nx*ny))
    K = np.zeros((DOF*nx*ny, DOF*nx*ny))
    plates = []
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        plate = BFSPlate2D()
        plate.n1 = n1
        plate.n2 = n2
        plate.n3 = n3
        plate.n4 = n4
        plate.ABD = lam.ABD
        update_K(plate, nid_pos, ncoords, K)
        plates.append(plate)

    # locating nodes
    bk = np.zeros(K.shape[0], dtype=bool) # constrained DOFs, can be used to prescribe displacements

    x = ncoords[:, 0]
    y = ncoords[:, 1]

    # applying boundary conditions
    # simply supported
    check = isclose(x, 0) | isclose(x, a) | isclose(y, 0) | isclose(y, b)
    bk[2::DOF] = check
    # point supports
    check = isclose(x, a/2) & (isclose(y, 0) | isclose(y, b))
    bk[0::DOF] = check
    check = isclose(y, b/2) & (isclose(x, 0) | isclose(x, a))
    bk[1::DOF] = check

    # unconstrained nodes
    bu = ~bk # logical_not

    # defining external force vector
    fext = np.zeros(K.shape[0], dtype=float)

    # applying unitary load along u at x=a
    # nodes at vertices get 1/2 the force
    nint = 9
    points, weights = get_points_weights(nint)
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
            fe += ley/2*weights[j]*plate.Nu(xi, eta)*Nxx
        fext[indices] += fe

    Kuu = K[bu, :][:, bu]
    fextu = fext[bu]

    # static solver
    uu = solve(Kuu, fextu)
    u = np.zeros(K.shape[0], dtype=float)
    u[bu] = uu

    if plot_static:
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
        update_Kg(u, plate, nid_pos, points, weights, Kg)
    Kguu = Kg[bu, :][:, bu]

    # solving modified generalized eigenvalue problem
    # Original: (K + lambda*KG)*v = 0
    # Modified: (-1/lambda)*K*v = KG*v  #NOTE here we find (-1/lambda)
    eigvals, eigvecsu = eigh(a=Kguu, b=Kuu, type=1)
    eigvals[eigvals !=0] = -1./eigvals[eigvals != 0]
    eigvecs = np.zeros_like(K)
    eigvecs[bu, :][:, bu] = eigvecsu

    if plot_lb:
        plt.gca().set_aspect('equal')
        mode = 0
        wplot = eigvecs[2::DOF, mode].reshape(nx, ny).T
        levels = np.linspace(wplot.min(), wplot.max(), 300)
        plt.contourf(xmesh, ymesh, wplot, levels=levels)
        plt.colorbar()
        plt.show()

    kc = 4
    Nxxcr_theoretical = E*np.pi**2*(h/b)**2*kc/(12*(1 - nu**2))*h
    print(Nxxcr_theoretical, eigvals[0])
    print(eigvals[0]/(E*np.pi**2*(h/b)**2/(12*(1 - nu**2))*h))
    assert isclose(eigvals[0], Nxxcr_theoretical, rtol=0.05)


if __name__ == '__main__':
    test_linear_buckling_iso_SSSS(plot_static=False, plot_lb=False, nx=15, ny=15)
