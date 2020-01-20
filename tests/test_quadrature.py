import numpy as np

from bfsplate2d.quadrature import get_points_weights

def test_quadrature():
    func = lambda xi: xi**3 + xi**2
    func_int_an = lambda xi: 1/4*xi**4 + 1/3*xi**3
    int_an = func_int_an(+1) - func_int_an(-1)
    for nint in range(2, 11):
        points, weights = get_points_weights(nint)
        int_num = 0
        for xi, wi in zip(points, weights):
            int_num += wi*func(xi)
        assert np.isclose(int_an, int_num)


if __name__ == '__main__':
    test_quadrature()


