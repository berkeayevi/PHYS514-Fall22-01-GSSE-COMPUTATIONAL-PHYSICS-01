import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import scipy.constants as cons

global G, solarm, c
G = 6.6743e-11
solarm = 1.98847e30
c = 3e8


def PtoGU(P):
    # Pressure in GU units
    return P * (((G ** 3) * (solarm ** 2)) / (c ** 8))


def dformGU(d):
    # d conversion form GU unit
    return d / ((G ** 3) * (solarm ** 2) / (c ** 6))


def RfromGU(R):
    # R conversion form GU unit
    return R * ((G * solarm) / (c ** 2))


def polyconvert(K, n, p):
    # Polytopic conversion
    return (np.abs(p) / K) ** (n / (n + 1))


def rhocal(K, n, P):

    pgu = PtoGU(P)
    pconv = polyconvert(K, n, pgu)
    return dformGU(pconv)


def tov_equation(r, s, K, n):

    m, v, p, m_p = s
    dens = polyconvert(K, n, p)
    mafter = 4 * np.pi * (r ** 2) * dens

    dpr = (r * (r - 2 * m))
    if dpr == 0:
        v_next = 0
    else:
        v_next = 2 * ((m + 4 * np.pi * (r ** 3) * p) / dpr)
    p_next = -0.5 * (p + dens) * v_next
    if r == 0:
        m_p_next = 0.0
    else:
        m_p_next = 4 * np.pi * ((1 - ((2 * m) / (r))) ** (-0.5)) * (r ** 2) * dens

    return np.array([mafter, v_next, p_next, m_p_next])


def TOV(pcent, K, n):
    # TOV solver IVP
    pg = PtoGU(pcent)
    # Stopping contidion
    def stop(r, s, K, n): return s[-2] - 1e-100
    stop.terminal = True

    s_0 = [0, 0, pg, 0] # Initial condition
    r_span = [0, 1e15] # span
    r = solve_ivp(tov_equation, r_span, s_0, events=stop, args=(K, n), max_step=1)

    return r.t[-1], r.y[0, -1], r.y[-1, -1]


def FinalQ2main():
    N1 = 20

    K = np.arange(100 - N1, 100 + N1 + 1, 1)
    N = K.shape[0]

    P = np.logspace(15, 50, N)
    n = 1
    mass = np.zeros((N, N))
    massp = np.zeros((N, N))
    r = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            rsoln, msoln, msolnp = TOV(P[j], K[i], n)

            mass[i][j] = msoln
            massp[i][j] = msolnp
            r[i][j] = rsoln

    Delm = (massp - mass) / mass  # equation 17
    rho = rhocal(K[N1], n, P)

    Mpc = np.gradient(mass[N1, :], rho)
    stable = np.nonzero(Mpc < 0)[0]
    unstable = np.nonzero(Mpc > 0)[0]

    R = RfromGU(r)  # in meter
    R = R / 1000  # in Km

    plt.figure()
    plt.plot(R[N1, :], mass[N1, :], 'x')
    plt.ylabel('Mass')
    plt.xlabel('Radius')
    plt.title('M vs R')

    plt.figure()
    plt.plot(R[N1, :], Delm[N1, :], 'x')
    plt.ylabel('del')
    plt.xlabel('Radius')
    plt.title('Del vs R')

    plt.figure()
    plt.yscale('log')
    plt.plot(mass[N1, :], rho[:], 'x')
    plt.xlabel('Mass')
    plt.ylabel('Rho')
    plt.title('Mass vs Rho')

    plt.figure()
    plt.plot(K, np.max(mass, axis=1))
    plt.xlabel('K')
    plt.ylabel('Max mass')
    plt.title('Max mass vs K')

    plt.show()

