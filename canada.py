import csv

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

x,y = np.loadtxt('canadajan25onwards.txt', unpack = True, delimiter = ',')

# Total population, N #
N = 37000000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.28, 1/7
# A grid of time points (in days)
t = np.linspace(0, 75)


# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T


fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

ax.plot(y, alpha=0.5, lw=2, label='ACTUAL')
ax.plot(t, I, 'g', alpha=0.5, lw=2, label='Nothing Done 1')


ax.set_xlabel('Time /days')
ax.set_ylabel('Number')
ax.set_ylim(0,200000)


ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()