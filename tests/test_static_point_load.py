import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from composites.laminate import read_stack

from bfsplate2d import (BFSPlate2D, update_KC0, DOF, KC0_SPARSE_SIZE, DOUBLE,
        INT)
from bfsplate2d.quadrature import get_points_weights

def test_static_point_load(plot=False):
    # number of nodes
    nx = 29
    ny = 29
    points, weights = get_points_weights(nint=4)

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

    num_elements = len(n1s)
    print('num_elements', num_elements)

    N = DOF*nx*ny
    Kr = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kc = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kv = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    init_k_KC0 = 0

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
        update_KC0(plate, points, weights, Kr, Kc, Kv)
        init_k_KC0 += KC0_SPARSE_SIZE
        plates.append(plate)

    KC0 = coo_matrix((Kv, (Kr, Kc)), shape=(N, N)).tocsc()

    # applying boundary conditions
    bk = np.zeros(KC0.shape[0], dtype=bool)

    # simply supported
    check = isclose(x, 0) | isclose(x, a) | isclose(y, 0) | isclose(y, b)
    bk[2::DOF] = check

    # eliminating all u,v displacements
    bk[0::DOF] = True
    bk[1::DOF] = True

    bu = ~bk # same as np.logical_not, defining unknown DOFs

    # external force vector for point load at center
    f = np.zeros(KC0.shape[0])
    fmid = 1.

    # force at center node
    check = np.isclose(x, a/2) & np.isclose(y, b/2)
    f[2::DOF][check] = fmid
    assert f.sum() == fmid

    # sub-matrices corresponding to unknown DOFs
    Kuu = KC0[bu, :][:, bu]
    fu = f[bu]

    # solving
    uu = spsolve(Kuu, fu)
    u = np.zeros(KC0.shape[0], dtype=float)
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

if __name__ == '__main__':
    test_static_point_load(plot=True)
