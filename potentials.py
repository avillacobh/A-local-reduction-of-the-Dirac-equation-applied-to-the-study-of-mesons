import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# =========================
# Parámetros
# =========================

alpha = 1.0

Vv_bar = 0.0
Vs_bar = 3.5

d_v = 0.4
d_s = 0.4

r_s = 2.0

# =========================
# Dominio radial
# =========================

r = np.linspace(0.01, 6, 2000)

# =========================
# Potenciales
# =========================

Vv = Vv_bar - (4 * alpha / 3) * erf(r / d_v) / r

Vs = 0.5 * Vs_bar * (erf((r - r_s) / d_s) - 1)

# =========================
# Derivadas analíticas
# =========================

dVv = (4 * alpha / 3) * (
    erf(r / d_v) / r**2
    - (2 / (np.sqrt(np.pi) * d_v * r))
      * np.exp(-(r / d_v)**2)
)

dVs = (
    Vs_bar / (np.sqrt(np.pi) * d_s)
    * np.exp(-((r - r_s) / d_s)**2)
)

# =========================
# Gráficas
# =========================

fig, ax = plt.subplots(2, 1, figsize=(8, 8))

# Potenciales
ax[0].plot(r, Vv, label=r"$V_v(r)$")
ax[0].plot(r, Vs, label=r"$V_s(r)$")

ax[0].set_ylabel("Potential")
ax[0].set_title("Potentials")
ax[0].grid(True)
ax[0].legend()

# Derivadas
ax[1].plot(r, dVv, label=r"$V_v'(r)$")
ax[1].plot(r, dVs, label=r"$V_s'(r)$")

ax[1].set_xlabel(r"$r$")
ax[1].set_ylabel("Derivative")
ax[1].set_title("Analytical derivatives")
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.show()