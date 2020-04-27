from multiprocessing import Pool
import numpy as np
import sympy

cpu_count = 6
DOF = 6


def integrate_simplify(integrand):
    return sympy.simplify(sympy.integrate(integrand, ('xi', -1, 1), ('eta', -1, 1)))


def main():
    sympy.var('xi, eta, lex, ley, h')

    ONE = sympy.Integer(1)

    # shape functions
    # - from Reference:
    #     OCHOA, O. O.; REDDY, J. N. Finite Element Analysis of Composite Laminates. Dordrecht: Springer, 1992.
    # bi-linear
    Li = lambda xii, etai: ONE/4.*(1 + xi*xii)*(1 + eta*etai)
    # cubic
    Hwi = lambda xii, etai: ONE/16.*(xi + xii)**2*(xi*xii - 2)*(eta+etai)**2*(eta*etai - 2)
    Hwxi = lambda xii, etai: -lex/32.*xii*(xi + xii)**2*(xi*xii - 1)*(eta + etai)**2*(eta*etai - 2)
    Hwyi = lambda xii, etai: -ley/32.*(xi + xii)**2*(xi*xii - 2)*etai*(eta + etai)**2*(eta*etai - 1)
    Hwxyi = lambda xii, etai: lex*ley/64.*xii*(xi + xii)**2*(xi*xii - 1)*etai*(eta + etai)**2*(eta*etai - 1)

    # node 1 (-1, -1)
    # node 2 (+1, -1)
    # node 3 (+1, +1)
    # node 4 (-1, +1)

    Nw = sympy.Matrix([[
       #u, v, w,     , dw/dx  , dw/dy,  d2w/(dxdy)
        0, 0, Hwi(-1, -1), Hwxi(-1, -1), Hwyi(-1, -1), Hwxyi(-1, -1),
        0, 0, Hwi(+1, -1), Hwxi(+1, -1), Hwyi(+1, -1), Hwxyi(+1, -1),
        0, 0, Hwi(+1, +1), Hwxi(+1, +1), Hwyi(+1, +1), Hwxyi(+1, +1),
        0, 0, Hwi(-1, +1), Hwxi(-1, +1), Hwyi(-1, +1), Hwxyi(-1, +1),
        ]])

    BA = (2/lex)*Nw.diff(xi)

    # mass matrix
    #TODO note that OFFSET is not supported
    KAe = sympy.zeros(4*DOF, 4*DOF)
    KAe[:, :] = (lex*ley)/4.*(Nw.T*BA)

    print('Integrating KAe')
    integrands = []
    for ind, integrand in np.ndenumerate(KAe):
        integrands.append(integrand)
    p = Pool(cpu_count)
    a = list(p.map(integrate_simplify, integrands))
    for i, (ind, integrand) in enumerate(np.ndenumerate(KAe)):
        KAe[ind] = a[i]

    # KA represents the aerodynamic stiffness matrix
    # in case we want to apply coordinate transformations
    KA = KAe

    def name_ind(i):
        if i >=0 and i < DOF:
            return 'c1'
        elif i >= DOF and i < 2*DOF:
            return 'c2'
        elif i >= 2*DOF and i < 3*DOF:
            return 'c3'
        elif i >= 3*DOF and i < 4*DOF:
            return 'c4'
        else:
            raise

    print('printing code for sparse implementation')
    for ind, val in np.ndenumerate(KA):
        if val == 0:
            continue
        print('                k += 1')
        print('                KAv[k] +=', KA[ind])

    print()
    print()
    print()
    KA_SPARSE_SIZE = 0
    for ind, val in np.ndenumerate(KA):
        if val == 0:
            continue
        KA_SPARSE_SIZE += 1
        i, j = ind
        si = name_ind(i)
        sj = name_ind(j)
        print('        k += 1')
        print('        KAr[k] = %d+%s' % (i%DOF, si))
        print('        KAc[k] = %d+%s' % (j%DOF, sj))
    print('KA_SPARSE_SIZE', KA_SPARSE_SIZE)


if __name__ == '__main__':
    main()

