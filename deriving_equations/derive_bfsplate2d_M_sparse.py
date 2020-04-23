"""
Mass matrix for BFS plate 2D
"""
from multiprocessing import Pool
import numpy as np
import sympy

num_nodes = 4
cpu_count = 6
DOF = 6

def integrate_simplify(integrand):
    return sympy.simplify(sympy.integrate(integrand, ('xi', -1, 1), ('eta', -1, 1)))

def main():
    sympy.var('xi, eta, lex, ley, rho, h')
    sympy.var('A11, A12, A16, A22, A26, A66')
    sympy.var('B11, B12, B16, B22, B26, B66')
    sympy.var('D11, D12, D16, D22, D26, D66')

    ONE = sympy.Integer(1)

    # shape functions
    # - from Reference:
    #     OCHOA, O. O.; REDDY, J. N. Finite Element Analysis of Composite Laminates. Dordrecht: Springer, 1992.
    # bi-linear
    Li = lambda xii, etai: ONE/4.*(1 + xi*xii)*(1 + eta*etai)
    # cubic
    Hw_i = lambda xii, etai: ONE/16.*(xi + xii)**2*(xi*xii - 2)*(eta+etai)**2*(eta*etai - 2)
    Hwx_i = lambda xii, etai: -lex/32.*xii*(xi + xii)**2*(xi*xii - 1)*(eta + etai)**2*(eta*etai - 2)
    Hwy_i = lambda xii, etai: -ley/32.*(xi + xii)**2*(xi*xii - 2)*etai*(eta + etai)**2*(eta*etai - 1)
    Hwxy_i = lambda xii, etai: lex*ley/64.*xii*(xi + xii)**2*(xi*xii - 1)*etai*(eta + etai)**2*(eta*etai - 1)

    # node 1 (-1, -1)
    # node 2 (+1, -1)
    # node 3 (+1, +1)
    # node 4 (-1, +1)

    Nu = sympy.Matrix([[
       #u, v, w,     , dw/dx  , dw/dy,  d2w/(dxdy)
        Li(-1, -1), 0, 0, 0, 0, 0,
        Li(+1, -1), 0, 0, 0, 0, 0,
        Li(+1, +1), 0, 0, 0, 0, 0,
        Li(-1, +1), 0, 0, 0, 0, 0,
        ]])
    Nv = sympy.Matrix([[
       #u, v, w,     , dw/dx  , dw/dy,  d2w/(dxdy)
        0, Li(-1, -1), 0, 0, 0, 0,
        0, Li(+1, -1), 0, 0, 0, 0,
        0, Li(+1, +1), 0, 0, 0, 0,
        0, Li(-1, +1), 0, 0, 0, 0,
        ]])
    Nw = sympy.Matrix([[
       #u, v, w,     , dw/dx  , dw/dy,  d2w/(dxdy)
        0, 0, Hw_i(-1, -1), Hwx_i(-1, -1), Hwy_i(-1, -1), Hwxy_i(-1, -1),
        0, 0, Hw_i(+1, -1), Hwx_i(+1, -1), Hwy_i(+1, -1), Hwxy_i(+1, -1),
        0, 0, Hw_i(+1, +1), Hwx_i(+1, +1), Hwy_i(+1, +1), Hwxy_i(+1, +1),
        0, 0, Hw_i(-1, +1), Hwx_i(-1, +1), Hwy_i(-1, +1), Hwxy_i(-1, +1),
        ]])
    # bending
    Nphix = -(2/lex)*Nw.diff(xi)
    Nphiy = -(2/ley)*Nw.diff(eta)

    #TODO
    # - offset for monolithic
    # - multi-material laminated shells (sandwhich etc)
    # - properties per integration point
    Me = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
    Me[:, :] = (lex*ley)/4.*rho*(h*(Nu.T*Nu + Nv.T*Nv + Nw.T*Nw)
          + h**3/12*(Nphix.T*Nphix + Nphiy.T*Nphiy))

    print('Integrating Me')
    integrands = []
    for ind, integrand in np.ndenumerate(Me):
        integrands.append(integrand)
    p = Pool(cpu_count)
    a = list(p.map(integrate_simplify, integrands))
    for i, (ind, integrand) in enumerate(np.ndenumerate(Me)):
        Me[ind] = a[i]

    # M represents the global mass matrix
    # in case we want to apply coordinate transformations
    M = Me

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
    for ind, val in np.ndenumerate(M):
        if val == 0:
            continue
        print('                k += 1')
        print('                Mv[k] +=', M[ind])

    print()
    print()
    print()
    M_SPARSE_SIZE = 0
    for ind, val in np.ndenumerate(M):
        if val == 0:
            continue
        M_SPARSE_SIZE += 1
        i, j = ind
        si = name_ind(i)
        sj = name_ind(j)
        print('        k += 1')
        print('        Mr[k] = %d+%s' % (i%DOF, si))
        print('        Mc[k] = %d+%s' % (j%DOF, sj))
    print('M_SPARSE_SIZE', M_SPARSE_SIZE)

if __name__ == '__main__':
    main()
