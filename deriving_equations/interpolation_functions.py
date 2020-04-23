import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def Hw(xi, w1, wr1, w2, wr2):
    lex = 2
    xi1, xi2 = -1, +1
    Hw_1 = -1/4.*(xi + xi1)**2*(xi*xi1 - 2)
    Hw_2 = -1/4.*(xi + xi2)**2*(xi*xi2 - 2)
    Hwx_1 = lex/32.*xi1*(xi + xi1)**2*(xi*xi1 - 1)
    Hwx_2 = lex/32.*xi2*(xi + xi2)**2*(xi*xi2 - 1)
    return Hw_1*w1 + Hwx_1*wr1 + Hw_2*w2 + Hwx_2*wr2

def Hw_xi(xi, w1, wr1, w2, wr2):
    lex = 2
    xi1, xi2 = -1, +1
    Hw_1 = -1/4.*(2*(xi + xi1)*(xi*xi1 - 2) + (xi + xi1)**2*xi1)
    Hw_2 = -1/4.*(2*(xi + xi2)*(xi*xi2 - 2) + (xi + xi2)**2*xi2)
    Hwx_1 = lex/8.*xi1*(2*(xi + xi1)*(xi*xi1 - 1) + (xi + xi1)**2*xi1)
    Hwx_2 = lex/8.*xi2*(2*(xi + xi2)*(xi*xi2 - 1) + (xi + xi2)**2*xi2)
    return Hw_1*w1 + Hwx_1*wr1 + Hw_2*w2 + Hwx_2*wr2

w1 = 1
w2 = 3
wr1 = 0.5
wr2 = 1
xi = np.linspace(-1, +1, 1000)
plt.plot(xi, Hw(xi, w1, wr2, w2, wr2))
plt.plot(xi, Hw_xi(xi, w1, wr1, w2, wr2))
plt.show()


