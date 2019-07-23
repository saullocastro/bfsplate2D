import sys
sys.path.append(r'C:\repositories\bfsplate2d')

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh, spsolve
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from composites.laminate import read_stack

from bfsplate2d import BFSPlate2D, update_K, update_Kg
from bfsplate2d.quadrature import get_points_weights

DOF = 6
def test_linear_buckling(plot_static=False, plot_lb=False):
    # number of nodes
    nx = 23
    ny = 23

    # geometry
    a = 1.0
    b = 0.5

    # material properties
    E = 0.7e11
    nu = 0.3
    lam = read_stack(stack=[0], plyt=0.001, laminaprop=[E, E, nu])

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


    # defining external force vector
    fext = np.zeros(K.shape[0], dtype=float)

    # applying unitary load along u at x=a
    # nodes at vertices get 1/2 the force
    Nxx = -1
    ftotal = Nxx*b # ny -> number of nodes along y
    check = (np.isclose(x, a) & ~np.isclose(y, 0) & ~np.isclose(y, b))
    fext[0::DOF][check] = ftotal/(ny - 1)
    check = ((np.isclose(x, a) & np.isclose(y, 0))
            |(np.isclose(x, a) & np.isclose(y, b)))
    fext[0::DOF][check] = ftotal/(ny - 1)/2

    Kuu = K[bu, :][:, bu]
    fextu = fext[bu]

    # static solver
    Kuu = csc_matrix(Kuu) # making Kuu a sparse matrix
    uu = spsolve(Kuu, fextu)
    u = np.zeros(K.shape[0], dtype=float)
    u[bu] = uu

    print('u extremes', u.min(), u.max())
    if plot_static:
        plt.gca().set_aspect('equal')
        uplot = u[0::DOF].reshape(nx, ny).T
        levels = np.linspace(uplot.min(), uplot.max(), 300)
        plt.contourf(xmesh, ymesh, uplot, levels=levels)
        plt.colorbar()
        plt.show()

    # eigenvalue solver

    # getting integration points
    points, weights = get_points_weights(nint=4)
    for plate in plates:
        update_Kg(u, plate, nid_pos, points, weights, Kg)
    Kguu = Kg[bu, :][:, bu]
    Kguu = csc_matrix(Kguu) # making Kguu a sparse matrix

    # solving modified generalized eigenvalue problem
    # Original: (K + lambda*KG)*v = 0
    # Modified: (-1/lambda)*K*v = KG*v  #NOTE here we find (-1/lambda)
    num_eigenvalues = 5
    print('eig solver begin')
    eigvals, eigvecsu = eigsh(A=Kguu, k=num_eigenvalues, which='SM', M=Kuu,
            tol=1e-6, sigma=1., mode='cayley')
    print('eig solver end')
    eigvals = -1./eigvals
    eigvecs = np.zeros((K.shape[0], num_eigenvalues), dtype=float)
    eigvecs[bu, :] = eigvecsu

    if plot_lb:
        plt.gca().set_aspect('equal')
        mode = 0
        wplot = eigvecs[2::DOF, mode].reshape(nx, ny).T
        levels = np.linspace(wplot.min(), wplot.max(), 300)
        plt.contourf(xmesh, ymesh, wplot, levels=levels)
        plt.colorbar()
        plt.show()
    print(eigvals)


if __name__ == '__main__':
    test_linear_buckling(plot_static=True, plot_lb=True)
