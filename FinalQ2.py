import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

global leng, smass
leng = 1477
smass = 1.989e30


def polyconvert(p, K, n):
    rho = (p / K) ** (n / (n + 1))
    return rho


def rhoconv(rho_c):
    return rho_c * (smass / (leng ** 3))


def rconv(r):
    return r * (leng / 1000)


def TOVeqn(r, y, K, n):
    m, v, p, mb = y

    dmdr = 4 * np.pi * (r ** 2) * polyconvert(p, K, n)
    dvdr = (2 * (m + 4 * np.pi * r ** 3 * p)) / (r * (r - 2 * m))
    dpdr = -0.5 * (p + polyconvert(p, K, n)) * dvdr
    dbmdr = 4 * np.pi * (1 - 2 * m / r) ** (-1 / 2) * r ** 2 * polyconvert(p, K, n)
    return [dmdr, dvdr, dpdr, dbmdr]


def TOV_sol(rho_c, K, n):
    def stop(r, y, K, n): return y[2]

    stop.terminal = True

    dt = 1e-10
    p_c = rho_c ** 2 * K
    trange = np.linspace(0 + dt, 100, 100)
    y_0 = [0, 0, p_c, 0]
    t_0 = [trange[0], trange[-1]]
    soln = solve_ivp(TOVeqn, t_0, y_0, args=(K, n), events=stop)

    return soln.t[-1], soln.y[0, -1], soln.y[-1, -1]


def FinalQ2main():
    N = 100
    K = 50
    n = 1

    rho_c = np.linspace(1e-1, 1e-5, N)

    r = np.zeros_like(rho_c)
    mass = np.zeros_like(rho_c)
    massb = np.zeros_like(rho_c)

    for i in range(len(rho_c)):
        solr, solm, solmb = TOV_sol(rho_c[i], K, n)

        r[i] = solr
        mass[i] = solm
        massb[i] = solmb

    r = rconv(r)  # to km
    rho_c = rhoconv(rho_c)
    delta = (massb - mass) / mass  # Delta calculation equation 17

    dmdrho = np.zeros_like(mass)
    for i in range(len(mass) - 1):
        dmdrho[i] = (mass[i + 1] - mass[i]) / (rho_c[i + 1] - rho_c[i])

    stable = np.nonzero(dmdrho > 0)[0]  # Equation 17
    unstable = np.nonzero(dmdrho < 0)[0]  # Equation 18

    plt.figure()
    plt.plot(r, mass)
    plt.title("Mass-Radius")
    plt.xlabel("Radius(km)")
    plt.ylabel("Mass")

    plt.figure()
    plt.plot(delta, r)
    plt.xlabel("Delta")
    plt.ylabel("Radius(km)")
    plt.title("Delta - Radius")

    plt.figure()
    plt.plot(rho_c, mass)
    plt.title("rho_c - mass")
    plt.xlabel("rho_c")
    plt.ylabel("mass")

    plt.figure()
    plt.plot(r[stable], mass[stable], '-.', label='Stable Regime')
    plt.plot(r[unstable], mass[unstable], '--', label='UnStable Regime')
    plt.title("Stability")
    plt.xlabel("Radius(km)")
    plt.ylabel("Mass(Solar Mass)")
    plt.legend()

    plt.show()


FinalQ2main()
