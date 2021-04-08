import astropy.constants as c
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import pdb

N = 10**15 * u.cm**-2
b = (6 * u.km/u.s).cgs
x = np.linspace(-5, 5) * (b/c.c.cgs).value
flam = np.logspace(-9, -4.89)

def tau(x, flam):
    prefac = np.sqrt(np.pi)*c.e.esu**2/c.m_e.cgs/c.c.cgs
    return (prefac * flam * N / b * np.exp(-(c.c.cgs*x/b)**2)).value

def W(x, flam):
    dx = np.diff(x)[0]
    sol_array = np.zeros(np.shape(flam))
    for i, f in enumerate(flam):
        sol_array[i] = np.sum((1 - np.exp(-tau(x, f)))/(x+1)**2*dx)
    return sol_array

plt.plot(flam, W(x, flam), marker='o')
plt.xlabel('$f\lambda_0$')
plt.ylabel('$W(f\lambda_0) / \lambda_0$')
plt.title('N={:.1e}, b={:.1e}'.format(N, b))
plt.tight_layout()
plt.show()


data = np.loadtxt('cog.data.txt', skiprows=1)
lam, f, wlam1, wlam1_err, wlam2, wlam2_err = data.T * u.angstrom.to(u.cm)
f = f/u.angstrom.to(u.cm)

plt.plot(flam, W(x, flam), marker='o', label='Model', alpha=0.5)
plt.errorbar(f*lam, wlam1/lam, yerr=wlam1_err, marker='^', linestyle='None', label='Wlam1')
plt.errorbar(f*lam, wlam2/lam, yerr=wlam2_err, marker='s', linestyle='None', label='Wlam2')
plt.xlabel('$f\lambda_0$')
plt.ylabel('$W(f\lambda_0) / \lambda_0$')
plt.title('N={:.1e}, b={:.1e}'.format(N, b))
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.show()

