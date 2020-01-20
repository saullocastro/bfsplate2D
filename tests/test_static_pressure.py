import sys
sys.path.append(r'C:\repositories\bfsplate2d')

import numpy as np
from numpy import isclose
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from composites.laminate import read_stack

from bfsplate2d import BFSPlate2D, update_K
from bfsplate2d.quadrature import get_points_weights

DOF = 6
def test_static_pressure(plot=False):
    # number of nodes
    nx = 5
    ny = 7

    # geometry
    a = 3
    b = 7

    # material properties
    E = 200e9
    nu = 0.3
    h = 0.005
    lam = read_stack(stack=[0], plyt=h, laminaprop=[E, E, nu])

    xtmp = np.linspace(0, a, nx)
    ytmp = np.linspace(0, b, ny)
    xmesh, ymesh = np.meshgrid(xtmp, ytmp)

    # getting nodes
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T
    x = ncoords[:, 0]
    y = ncoords[:, 1]
    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    nids_mesh = nids.reshape(nx, ny)

    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

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

    # applying boundary conditions
    bk = np.zeros(K.shape[0], dtype=bool)

    # simply supported
    check = isclose(x, 0) | isclose(x, a) | isclose(y, 0) | isclose(y, b)
    bk[2::DOF] = check
    # point supports (middle)
    check = isclose(x, a/2) & isclose(y, b/2)
    bk[0::DOF] = check
    bk[1::DOF] = check

    bu = ~bk

    # external force vector applying consistent pressure
    f = np.zeros(K.shape[0], dtype=float)
    P = 10 # Pa
    nint = 4
    points, weights = get_points_weights(nint)
    for plate in plates:
        pos1 = nid_pos[plate.n1]
        pos2 = nid_pos[plate.n2]
        pos3 = nid_pos[plate.n3]
        pos4 = nid_pos[plate.n4]
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
        for i in range(nint):
            xi = points[i]
            weight_xi = weights[i]
            for j in range(nint):
                eta = points[j]
                weight_eta = weights[j]
                weight = weight_xi*weight_eta
                fe += (lex*ley)/4*weight*plate.Nw(xi, eta)*P
        f[indices] += fe

    Kuu = K[bu, :][:, bu]
    fu = f[bu]

    # solving
    Kuu = csc_matrix(Kuu) # making Kuu a sparse matrix
    uu = spsolve(Kuu, fu)
    u = np.zeros(K.shape[0], dtype=float)
    u[bu] = uu

    w = u[2::DOF].reshape(nx, ny).T
    print('wmax', w.max())
    print('wmin', w.min())
    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        levels = np.linspace(w.min(), w.max(), 300)
        plt.contourf(xmesh, ymesh, w, levels=levels)
        plt.colorbar()
        plt.show()
    assert np.isclose(w.max(), 3.9407e-3, rtol=1e-3)

if __name__ == '__main__':
    test_static_pressure(plot=True)
