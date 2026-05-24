import numpy as np
import matplotlib.pyplot as plt
import mpmath
import mittag_leffler

#precision depth
mpmath.mp.dps = 15

#params
a = 0.9
b = 0.7
c = 1

t_vals = np.linspace(1e-4, 35, 100)

#Mittag-Leffler function as in 1.36j
ML = np.zeros_like(t_vals)
for i, t in enumerate(t_vals):
    z = -c * t**a
    ML[i] = (t**(b - 1)) * mittag_leffler.mittag_leffler(z, a, b)

#Laplace transform of the Mittag-Leffler function we will test
def ML_hat(s):
    return s**(a - b) / (s**a + c)

#talbot method for laplace inversion
inversion = np.zeros_like(t_vals)
for i, t in enumerate(t_vals):
    if t <= 0:
        raise ValueError("must obey t > 0")
    else:
        inversion[i] = float(mpmath.invertlaplace(ML_hat, t, method = "talbot"))

plt.plot(t_vals, ML, color = "red", linewidth = 2, label = "mittag_leffler package")
plt.plot(t_vals, inversion, "g", linewidth = 1, label = "Inverse Laplace transform")
plt.xlabel(r"$t$")
plt.ylabel(r"ML($t$)")
plt.legend()
plt.show()