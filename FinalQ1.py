import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import scipy.constants as cons


def readdata(filename):
    datafile = open(filename, 'r')  # opens the file
    name = []  # inilize the params
    logg = []
    mass = []
    lines = datafile.readlines()
    lines = lines[1:]  # crops the first line
    for i in lines:
        (a, b, c) = i.split(',')  # split by coma
        name.append(a)
        logg.append(float(b))
        mass.append(float(c))
    datafile.close()

    return np.array(logg), np.array(mass)


def data_conversation(mass, logg):
    # Data conversation form .cvs file
    G = 6.6743e-11  # Gravitational constant
    solarm = 1.98847e30  # Solar mass
    avEradi = 6371e3  # Average Earth Radius

    g = 10 ** (logg) * 1e-2

    masskg = solarm * mass  # mass in Kg
    Rm = np.sqrt(G * masskg / g)  # Radius
    RavEradi = Rm / avEradi

    return masskg, Rm, RavEradi  # mass in kg, R in meter, R average Earth Radii


def eqnmass(r, rho):
    return 4 * np.pi * (r ** 2) * rho  # Equation (1)


def eqnpres(r, m, rho):
    G = 6.6743e-11  # Gravitational constant
    if r == 0:
        return 0
    else:
        return -(G * m * rho) / (r ** 2)  # Equation (1)


def eqndens(r, rho, m, C, q, D):
    x = (rho / D) ** (1 / q)
    # Derivative of P wrt x eqn (8)
    s1 = ((2 * x ** 2 - 3) * ((x ** 2 + 1) ** 0.5))
    s2 = ((x) * (4 * x) * ((x ** 2 + 1) ** 0.5))
    s3 = ((x) * (2 * x ** 2 - 3) * (0.5) * (2 * x) * ((x ** 2 + 1) ** -0.5))
    s4 = ((3) / (np.sqrt(x ** 2 + 1)))
    dpdx = C * (s1 + s2 + s3 + s4)
    dxdrho = (D ** (-1 / q)) * (rho ** ((1 / q) - 1)) / (q)
    dpdr = eqnpres(r, m, rho)

    if dpdx == 0 or dxdrho == 0:
        return 0
    else:
        return dpdr / (dpdx * dxdrho)


def laneEmden(ksi, teta, n):
    if ksi == 0:
        return [teta[1], 0]
    else:
        return np.array([teta[1], ((-ksi * (teta[0] ** n)) - 2 * teta[1]) / ksi])  # Equation(4)


def solnLaneEmden(n):
    teta = [1, 0]  # Initial condition
    ksi = [0, 10]  # range of the ksi

    soln = solve_ivp(laneEmden, ksi, teta, args=(n,))  # solve lane-Emden Equation

    return soln.t, soln.y


def MassProbR(R, C, q):
    n = q / (5 - q)

    mass = C * (R ** ((3 - n) / (1 - n)))  # Equation 7

    return mass


def CtoK(C, q):  # Conversion K from M = C *R
    G = 6.6743e-11  # Gravitational constant

    n = q / (5 - q)

    t, y = solnLaneEmden(n)
    ts = t[-1]
    yprime = y[1, -1]

    U = (((n + 1) / (4 * np.pi * G)) ** (((n - 3) / (2 - 2 * n)) + 1.5)) * (
            (ts) ** (((n - 3) / (1 - n)) + 2)) * 4 * np.pi * (-yprime)
    K = (C / U) ** ((n - 1) / (n))

    return K


def CDqtoK(q):
    # Calculation of C and D given in equation 11
    me = cons.m_e
    c = cons.c
    hbar = cons.hbar
    mu = cons.m_u
    pi = np.pi
    nue = 2

    C = ((me ** 4) * (c ** 5)) / (24 * (pi ** 2) * (hbar ** 3))

    D = ((me ** 3) * (mu) * (c ** 3) * (nue)) / (3 * (pi ** 2) * (hbar ** 3))
    # print('C = '+str(C) + ' \nD = '+ str(D)) # printing the value

    return (8 * C) / (5 * (D ** (5 / q))), C, D


def RKqMassRrelation(R, K, q):  # The relation when R, K and q is know
    G = 6.6743e-11  # Gravitational constant
    n = q / (5 - q)
    t, y = solnLaneEmden(n)
    ts = t[-1]
    yprime = y[1, -1]

    c1 = (((K * (n + 1)) / (4 * np.pi * G)) ** (-0.5 * ((3 - n) / (1 - n))))
    c2 = (((1 / (4 * np.pi)) * (((4 * np.pi * G) / (K * (n + 1))) ** 1.5)
           * (1 / ((ts ** 2) * (-yprime)))) ** (-1)) * (ts ** ((n - 3) / (1 - n))) * (R ** ((3 - n) / (1 - n)))

    M = c1 * c2
    return M


def MKqdensityC(M, K, q):
    G = 6.6743e-11  # Gravitational constant
    n = q / (5 - q)

    t, y = solnLaneEmden(n)
    ts = t[-1]
    yprime = y[1, -1]

    return ((M / (4 * np.pi)) * (((4 * np.pi * G) / (K * (n + 1))) ** 1.5) * (
            1 / ((ts ** 2) * (-yprime)))) ** ((2 * n) / (3 - n))


def MDE(r, m_dens, C, q, D):
    m = m_dens[0]
    dens_1 = m_dens[1]
    masseq = eqnmass(r, dens_1)
    DE = eqndens(r, dens_1, m, C, q, D)
    return [masseq, DE]


def MR(dens, q):
    K, C, D = CDqtoK(q)

    dens_0 = [0, dens]
    r_0 = [0, 1e8]
    r = solve_ivp(MDE, r_0, dens_0, args=(C, q, D))

    R = r.t[-1]
    M = r.y[0, -1]

    return R, M


def FinalQ1main():
    filename = 'white_dwarf_data.csv'  # reading .csv file
    logg, mass = readdata(filename)  # read data

    index = np.nonzero(mass < 0.34)[0]
    masskg, Rm, RavEradi = data_conversation(mass[index], logg[index])  # convert to usefull format

    solarm = 1.98847e30  # Solar mass
    q = 3
    C = 1e50

    y_0 = [C, q]

    fit, _ = curve_fit(MassProbR, Rm, masskg,y_0)
    print('nstar = ' + str(fit[1]) + '\nKstar= ' + str(fit[0]))
    K1 = CtoK(fit[0], round(fit[1]))

    Rkqmass = RKqMassRrelation(Rm, K1, q)
    cdens = MKqdensityC(Rkqmass, K1, q)

    M = np.zeros_like(cdens)
    R = np.zeros_like(cdens)
    for i in range(len(cdens)):
        M[i], R[i] = MR(cdens[i], q)

    plt.figure()
    plt.plot(RavEradi, mass[index], 'x')  # Ploting M vs R to converted data

    plt.xlabel('R(Average Earth Radius)')
    plt.ylabel('M (in solar mass)')
    plt.title('M vs R data plot')

    plt.figure()
    plt.yscale('log')
    plt.plot(RavEradi, masskg, 'x', label='WD data')
    plt.plot(RavEradi, Rkqmass, 'x', label='fit mass')
    plt.plot(RavEradi, R, 'x', label='Part E from given C,D and q')
    plt.xlabel('R(Average Earth Radius)')
    plt.ylabel('M (in solar mass) in log scale')
    plt.title('M(log) vs R data plot')
    plt.legend()

    plt.figure()
    plt.plot(mass[index], cdens, 'x')
    plt.xlabel('M (in solar mass)')
    plt.ylabel('central density')
    plt.title('central density data')

    plt.show()
FinalQ1main()