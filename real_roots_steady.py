import matplotlib.pyplot as plt
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

g = 1.0
r = 1.0
q2_limit = np.pi/2*.999
q2 = np.linspace(-q2_limit, q2_limit, 500)
q3dot_1 = -np.sqrt(20.0*g/9/r*np.sin(q2)*np.tan(q2))
q3dot_2 = np.sqrt(20.0*g/9/r*np.sin(q2)*np.tan(q2))
fig, ax = plt.subplots()

ax.fill_between(q2, q3dot_1, q3dot_2, facecolor='grey')
ax.set_xlabel(r'Lean $q_2$ (rad)')
ax.set_ylabel(r'Spin rate $\dot{q}_3$ (rad/s)')
ax.xaxis.set_ticks([-q2_limit, -np.pi/4, 0, np.pi/4, q2_limit])
ax.xaxis.set_ticklabels([r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"])
ax.yaxis.set_ticks([-2*q2_limit, -2*np.pi/4, 0, 2*np.pi/4, 2*q2_limit])
ax.yaxis.set_ticklabels([r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"])
ax.set_xbound(-q2_limit, q2_limit)
ax.set_ybound(-2*q2_limit, 2*q2_limit)
plt.text(-np.pi/8, np.pi/2, 'Equilibrium feasible')#, fontdict=font)
plt.text(-np.pi/8, -np.pi/2, 'Equilibrium feasible')#, fontdict=font)
plt.text(-3*np.pi/8, 0, 'Equilibrium infeasible')#, fontdict=font)
plt.text(np.pi/8, 0, 'Equilibrium infeasible')#, fontdict=font)
#plt.title("Feasibility regions for rolling disk, $m = r = g = 1$")
plt.show()

