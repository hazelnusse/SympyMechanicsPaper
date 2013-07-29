from sympy import (symbols, Matrix, eye, zeros, pi, trigsimp,
        solve_linear_system_LU, solve)
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point, dot,
cross, mprint, RigidBody, inertia, KanesMethod, mlatex)
from sympy import *
from sympy.physics.mechanics import *

# Symbols for time and constant parameters
t, r, m, g, v = symbols('t r m g v')

# Configuration variables and their time derivatives
# q1 -- yaw
# q2 -- lean
# q3 -- spin
# q4 -- disc center distance from inertial origin, N.x direction
# q5 -- disc center distance from inertial origin, N.y direction
# q6 -- disc center distance from inertial origin, N.z direction
q1, q2, q3, q4, q5, q6 = q = dynamicsymbols('q1:7')
q1d, q2d, q3d, q4d, q5d, q6d = qd = [qi.diff(t) for qi in q]
q_zero = {qi : 0 for qi in q}
qd_zero = {qdi : 0 for qdi in qd}

# Generalized speeds and their time derivatives
# u1 -- disc angular velocity component, disc fixed x direction
# u2 -- disc angular velocity component, disc fixed y direction
# u3 -- disc angular velocity component, disc fixed z direction
# u4 -- disc velocity component, disc fixed x direction
# u5 -- disc velocity component, disc fixed y direction
# u6 -- disc velocity component, disc fixed z direction
u = dynamicsymbols('u:6')
u1, u2, u3, u4, u5, u6 = u = dynamicsymbols('u1:7')
u1d, u2d, u3d, u4d, u5d, u6d = ud = [ui.diff(t) for ui in u]
u_zero = {ui : 0 for ui in u}
ud_zero = {udi : 0 for udi in ud}
ud_sym = dict(zip(ud, symbols('ud1:7')))
ud_sym_inv = {v: k for (k, v) in ud_sym.items()}

# Reference frames
N = ReferenceFrame('N')                   # Inertial frame
A = N.orientnew('A', 'Axis', [q1, N.z])   # Yaw intermediate frame
B = A.orientnew('B', 'Axis', [q2, A.x])   # Lean intermediate frame
C = B.orientnew('C', 'Axis', [q3, B.y])   # Disc fixed frame

# Inertial angular velocity and angular acceleration of disc fixed frame
C.set_ang_vel(N, u1*C.x + u2*C.y + u3*C.z)
C.set_ang_acc(N, u1d*C.x + u2d*C.y + u3d*C.z)
print("Angular acceleration")
mprint(C.ang_acc_in(N))

# Points
NO = Point('NO')                                       # Inertial origin
CO = Point('CO')                                       # Disc center
CO.set_vel(N, u4*C.x + u5*C.y + u6*C.z)
CO.set_acc(N, CO.vel(N).dt(C) + cross(C.ang_vel_in(N), CO.vel(N)))
print("Translational acceleration")
mprint(CO.acc(N))

P = CO.locatenew('P', r*B.z)                           # Disc-ground contact
P.v2pt_theory(CO, N, C)
P.a2pt_theory(CO, N, C)

# Configuration constraint and its Jacobian w.r.t. q        (Table 1)
f_c = Matrix([q6 - dot(CO.pos_from(P), N.z)])
f_c_dq = f_c.jacobian(q)
print("Configuration constraint")
mprint(f_c)

# Velocity level constraints                                (Table 1)
f_v = Matrix([dot(P.vel(N), uv) for uv in C])
f_v_dq = f_v.jacobian(q)
f_v_du = f_v.jacobian(u)
print("Velocity constraints")
mprint(f_v)

# Disc angular velocity in N expressed using time derivatives of coordinates
w_c_n_qd = q1d*A.z + q2d*B.x + q3d*B.y
# Disc center velocity in N expressed using time derivatives of coordinates
v_co_n_qd = q4d*N.x + q5d*N.y + q6d*N.z

# Kinematic differential equations
kindiffs = Matrix([dot(w_c_n_qd - C.ang_vel_in(N), uv) for uv in B] +
                  [dot(v_co_n_qd - CO.vel(N), uv) for uv in B])

print("Kinematic differential equations")
for kd_i in kindiffs:
    print("{0} &= 0 \\\\".format(msprint(kd_i)))

# f_0 and f_1                                               (Table 1)
f_0 = kindiffs.subs(u_zero)
f_1 = kindiffs.subs(qd_zero)

# f_0 == f_0_coef_matrix * qd, used to solve for qdots
f_0_coef_matrix = zeros((6,6))
for i in range(6):
    for j in range(6):
        f_0_coef_matrix[i, j] = f_0[i].diff(qd[j])

# Acceleration level constraints                            (Table 1)
f_a = f_v.diff(t)
print("Acceleration level constraints")
mprint(f_a)


# Kane's dynamic equations via elbow grease
# Partial angular velocities and velocities
partial_w_C = [C.ang_vel_in(N).diff(ui, N) for ui in u]
partial_v_CO = [CO.vel(N).diff(ui, N) for ui in u]

# Active forces
F_O = m*g*A.z
# Generalized active forces (unconstrained)
gaf = [dot(F_O, pv) for pv in partial_v_CO]
print("Generalized active forces (unconstrained)")
mprint(gaf)


# Inertia force
R_star_O = -m*CO.acc(N)

JJ = m * r**2 / 2
II = m * r**2 / 4

# Inertia torque
I_C_CO = inertia(C, II, JJ, II)     # Inertia of disc C about point CO
T_star_C = (-dot(I_C_CO, C.ang_acc_in(N))
            - cross(C.ang_vel_in(N), dot(I_C_CO, C.ang_vel_in(N))))

# Generalized inertia forces (unconstrained)
gif = [dot(R_star_O, pv) + dot(T_star_C, pav) for pv, pav in
        zip(partial_v_CO, partial_w_C)]
print("Generalized inertia forces (unconstrained)")
mprint(gif)

# Constrained dynamic equations

# Coordinates to be independent: q1, q2, q3, q4, q5
# Coordinates to be dependent: q6
# Already in the correct order, so permutation matrix is simply a 6x6 identity
# matrix
Pq = eye(6)
Pqi = Pq[:, :-1]
Pqd = Pq[:, -1]

# Speeds to be independent:  u1, u2, u3
# Speeds to be dependent:  u4, u5, u6
# Already in the correct order, so permutation matrix is simply a 6x6 identity
# matrix
Pu = eye(6)
Pui = Pu[:, :-3]
Pud = Pu[:, -3:]

# The constraints can be written as:
# Bi * ui + Bd * ud = 0
Bi = f_v_du*Pui
Bd = f_v_du*Pud
Bd_inv_Bi = -Bd.inverse_ADJ() * Bi

indep_indices = [0, 1, 2]   # Body fixed angular velocity measure numbers
dep_indices = [3, 4, 5]     # Body fixed velocity measure numbers

gif_indep = Matrix([gif[i] for i in indep_indices])
gif_dep = Matrix([gif[i] for i in dep_indices])
gaf_indep = Matrix([gaf[i] for i in indep_indices])
gaf_dep = Matrix([gaf[i] for i in dep_indices])

gif_con = gif_indep + Bd_inv_Bi.T * gif_dep
gaf_con = gaf_indep + Bd_inv_Bi.T * gaf_dep

print("Generalized inertia forces (constrained)")
mprint(gif_con)

print("Generalized activeforces (constrained)")
mprint(gaf_con)

# Build the part of f_2 and f_3 that come from Kane's equations, the first three
# rows of each
f_2 = zeros(3, 1)
f_3 = zeros(3, 1)
Muud = zeros (3, len(u))
for i in [0, 1, 2]:
    # Mass matrix terms and f2
    for j, udj in enumerate(ud):
        Muud[i, j] = gif_con[i].diff(udj)
        f_2[i] += Muud[i, j]*udj
        # Verify that there are no du/dt terms in the generalized active forces
        assert gaf_con[i].diff(udj) == 0
    # All other terms that don't have du/dt in them
    f_3[i] = gif_con[i].subs(ud_zero) + gaf_con[i]

print("f_c:")
mprint(f_c)
print("f_v:")
mprint(f_v)
print("f_a:")
mprint(f_a)
print("f_0:")
mprint(f_0)
print("f_1:")
mprint(f_1)
print("f_2:")
mprint(f_2)
print("f_3:")
mprint(f_3)


# Linearization Code

qdep = Pqd.T * Matrix(q)
udep = Pud.T * Matrix(u)

n = len(q)
l = len(qdep)
o = len(u)
#m = len(udep)

# Point of linearization
eq_point = {q1: 0,
            q2: 0,
            q3: 0,
            q4: 0,
            q5: 0,
            q6: -r,
#            u1: 0,
#            u2: -v/r,
#            u3: 0,
#            u4: v,
#            u5: 0,
#            u6: 0,
            u1d: 0,
            u2d: 0,
            u3d: 0}

print(f_a.subs(eq_point))
print((f_2 + f_3).subs(eq_point))
stop

udzero = dict(zip(ud, [0] * o))

M_qq = f_0.jacobian(qd)
M_uqc = f_a.jacobian(qd).subs(udzero)
M_uuc = f_a.jacobian(ud).subs(udzero)
M_uqd = f_2.jacobian(qd)
M_uud = f_2.jacobian(ud)
A_qq = -(f_0 + f_1).jacobian(q)
A_qu = -f_1.jacobian(u)
A_uqc = -f_a.jacobian(q).subs(udzero)
A_uuc = -f_a.jacobian(u).subs(udzero)
A_uqd = -(f_2 + f_3).jacobian(q)
A_uud = -f_3.jacobian(u)

f_c_jac_q = f_c.jacobian(q)
f_v_jac_q = f_v.jacobian(q)
f_v_jac_u = f_v.jacobian(u)
C_0 = (eye(n) - Pqd * (f_c_jac_q * Pqd).inv() * f_c_jac_q) * Pqi
C_1 = -Pud * (f_v_jac_u * Pud).inv() * f_v_jac_q
C_2 = (eye(o) - Pud * (f_v_jac_u * Pud).inv() * f_v_jac_u) * Pui

row1 = M_qq.row_join(zeros(n, o))
row2 = M_uqc.row_join(M_uuc)
row3 = M_uqd.row_join(M_uud)
M = row1.col_join(row2).col_join(row3)

M.simplify()
print("M:")
mprint(M)

row1 = ((A_qq + A_qu * C_1) * C_0).row_join(A_qu * C_2)
row2 = ((A_uqc + A_uuc * C_1) * C_0).row_join(A_uuc * C_2)
row3 = ((A_uqd + A_uud * C_1) * C_0).row_join(A_uud * C_2)
Amat = row1.col_join(row2).col_join(row3)

Amat.simplify()

Amat = Amat.applyfunc(lambda x: trigsimp(x.expand(), deep=True, recursive=True))

print("A:")
mprint(Amat)

# Now, selecting an equilibrium point to linearize about
"""
Known
-----
lean angle, q[1]*
spin rate, u[1]*
qd[0]
qd[1] = qd[5] = 0
ud[2] = ud[4] = 0

Solve for
---------
q[5]                                f_c
qd[2], qd[3], qd[4]                 
u[0 & 2:]                           
ud[0], ud[1], ud[3], ud[5]          
"""


subdict = {qd[1]:0, qd[5]:0, ud[2]:0, ud[4]:0, qd[0]:-(6 * qd[2]) / (5 * sin(q[1]))}
subdict.update({I : m / 4 * r**2, J : m / 2 * r**2})
kin_eq = kindiffs.subs(subdict)[3:, 0]
dyn_eq = f_a.col_join(f_2 + f_3).subs(subdict)

subdict.update(solve(f_c, q[5]))


qd_omega = (qd[0] * A.z + qd[2] * B.y).subs(subdict).express(C)
subdict.update({u[0]:(qd_omega & C.x)})
subdict.update({u[2]:(qd_omega & C.z)})
subdict.update(solve(Matrix([u[1] - (qd_omega & C.y)]), qd[2]))
subdict.update(solve(f_v.subs(subdict), u[3:]))


subdict.update(solve(kin_eq.subs(subdict), qd[3:5]))
for i in subdict.keys():
    subdict.update({i: simplify(sympify(subdict[i]).subs(subdict))})

dyn_eq = dyn_eq.subs(ud_sym).subs(subdict).subs(ud_sym_inv)
dyn_eq.simplify()

#udsol = solve(dyn_eq, ud)
dyn_eq_jacud = dyn_eq.jacobian([ud[0], ud[1], ud[3], ud[5]])
dyn_eq_ud = dyn_eq_jacud * Matrix([ud[0], ud[1], ud[3], ud[5]])
dyn_eq_qd = dyn_eq - dyn_eq_ud
temp = dyn_eq_jacud.row_join(-dyn_eq_qd)
temp.row_del(5)
temp.row_del(1)
udsol = solve_linear_system_LU(temp, [ud[0], ud[1], ud[3], ud[5]])

for i in udsol.keys():
    udsol.update({i : trigsimp(simplify(udsol[i].subs(subdict)), deep=True, recursive=True )})
subdict.update(udsol)
temp = subdict.pop(q[5])


M = M.subs(subdict).subs({q[5] : temp})
Amat = Amat.subs(subdict).subs({q[5] : temp})

M.simplify()
Amat.simplify()
Amat = Amat.applyfunc(lambda x: trigsimp(x.expand(), deep=True, recursive=True))

print('M:')
mpprint(M)
print('A:')
mpprint(Amat)


"""
stop

# Kane's dynamic equations via sympy.physics.mechanics
Bodies_List = [RigidBody('Disk', O, C, m, (inertia(C, I, J, I), O))]
Forces_List = [(O, m*g*A.z)]

KM = Kane(N)
KM.coords(q[:5], [q[5]], f_c)
KM.speeds(u[:3], u[3:], f_v)
KM.kindiffeq(kindiffs)
KM.kanes_equations(Forces_List, Bodies_List)
mm = KM.mass_matrix_full
mprint(mm)
f = KM.forcing_full
f.simplify()
mprint(f[-6:-9])
"""
