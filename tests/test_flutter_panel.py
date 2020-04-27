import sys
sys.path.append('..')

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs
import numpy as np
from composites.laminate import read_stack

from bfsplate2d import (BFSPlate2D, update_KC0, update_M, update_KA, DOF,
        KC0_SPARSE_SIZE, M_SPARSE_SIZE, KA_SPARSE_SIZE, DOUBLE, INT)
from bfsplate2d.quadrature import get_points_weights


def test_flutter_panel(plot=False):
    # number of nodes
    nx = 21 # along x
    ny = 11 # along y

    points, weights = get_points_weights(nint=4)

    # geometry
    a = 0.5
    b = 0.3

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
    KAr = np.zeros(KA_SPARSE_SIZE*num_elements, dtype=INT)
    KAc = np.zeros(KA_SPARSE_SIZE*num_elements, dtype=INT)
    KAv = np.zeros(KA_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    init_k_KC0 = 0
    init_k_M = 0
    init_k_KA = 0

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
        plate.init_k_KA = init_k_KA
        update_KC0(plate, points, weights, Kr, Kc, Kv)
        update_M(plate, Mr, Mc, Mv)
        update_KA(plate, KAr, KAc, KAv)
        init_k_KC0 += KC0_SPARSE_SIZE
        init_k_M += M_SPARSE_SIZE
        init_k_KA += KA_SPARSE_SIZE
        plates.append(plate)

    KC0 = coo_matrix((Kv, (Kr, Kc)), shape=(N, N)).tocsc()
    M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()
    KA = coo_matrix((KAv, (KAr, KAc)), shape=(N, N)).tocsc()

    # applying boundary conditions
    # simply supported

    # locating nodes
    bk = np.zeros(KC0.shape[0], dtype=bool) # constrained DOFs, can be used to prescribe displacements

    x = ncoords[:, 0]
    y = ncoords[:, 1]

    # constraining w at all edges
    check = (np.isclose(x, 0.) | np.isclose(x, a) | np.isclose(y, 0.) | np.isclose(y, b))
    bk[2::DOF] = check
    #NOTE uncomment for clamped
    #bk[3::DOF] = check
    #bk[4::DOF] = check
    # removing u,v
    bk[0::DOF] = True
    bk[1::DOF] = True

    # unconstrained nodes
    bu = ~bk # logical_not

    Kuu = KC0[bu, :][:, bu]
    Muu = M[bu, :][:, bu]
    KAuu = KA[bu, :][:, bu]

    num_eigenvalues = 10

    def MAC(mode1, mode2):
        return (mode1@mode2)**2/((mode1@mode1)*(mode2@mode2))

    MACmatrix = np.zeros((num_eigenvalues, num_eigenvalues))
    betastar = np.linspace(0, 250, 50)
    betas = betastar*E*h**3/a**3
    omegan_vec = []
    for i, beta in enumerate(betas):
        print('analysis i', i)
        # solving generalized eigenvalue problem
        eigvals, eigvecsu = eigs(A=Kuu + beta*KAuu, M=Muu,
                k=num_eigenvalues, which='LM', sigma=-1.)
        eigvecs = np.zeros((KC0.shape[0], num_eigenvalues), dtype=float)
        eigvecs[bu, :] = eigvecsu
        omegan_vec.append(eigvals**0.5)

        if i == 0:
            eigvecs_ref = eigvecs

        for j in range(num_eigenvalues):
            for k in range(num_eigenvalues):
                MACmatrix[j, k] = MAC(eigvecs_ref[:, j], eigvecs[:, k])
        print(np.round(MACmatrix, 1))

        eigvecs_ref = eigvecs.copy()


    omegan_vec = np.array(omegan_vec)

    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        for i in range(num_eigenvalues):
            plt.plot(betastar, omegan_vec[:, i])
        plt.show()


if __name__ == '__main__':
    test_flutter_panel(plot=True)
