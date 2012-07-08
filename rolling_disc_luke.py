from sympy import (symbols, Matrix, eye, zeros, pi, trigsimp,
solve_linear_system_LU, solve)
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point, dot,
cross, mprint, RigidBody, inertia, Kane, mlatex)
from sympy import *
from sympy.physics.mechanics import *

# Symbols for time and constant parameters
t, r, m, g, I, J = symbols('t r m g I J')

# Configuration variables and their time derivatives
# q[0] -- yaw
# q[1] -- lean
# q[2] -- spin
# q[3] -- disc center distance from inertial origin, N.x direction
# q[4] -- disc center distance from inertial origin, N.y direction
# q[5] -- disc center distance from inertial origin, N.z direction
q = dynamicsymbols('q:6')
qd = [qi.diff(t) for qi in q]
q_zero = {qi : 0 for qi in q}
qd_zero = {qdi : 0 for qdi in qd}

# Generalized speeds and their time derivatives
# u[0] -- disc angular velocity component, disc fixed x direction
# u[1] -- disc angular velocity component, disc fixed y direction
# u[2] -- disc angular velocity component, disc fixed z direction
# u[3] -- disc velocity component, disc fixed x direction
# u[4] -- disc velocity component, disc fixed y direction
# u[5] -- disc velocity component, disc fixed z direction
u = dynamicsymbols('u:6')
ud = [ui.diff(t) for ui in u]
u_zero = {ui : 0 for ui in u}
ud_zero = {udi : 0 for udi in ud}
ud_sym = dict(zip(ud, symbols('ud:6')))
ud_sym_inv = {v: k for (k, v) in ud_sym.items()}

# Reference frames
azi, ele, d = symbols('azi ele d')
cam = ReferenceFrame('cam')  # OpenGL camera frame, x right, y up
cam_p = cam.orientnew('cam_p', 'Axis', [-pi/2, cam.y])
ele_f = cam_p.orientnew('ele_f', 'Axis', [pi/2, cam_p.x])
azi_f = ele_f.orientnew('azi_f', 'Axis', [-ele, ele_f.y])
N = azi_f.orientnew('N', 'Axis', [-azi, azi_f.z])
A = N.orientnew('A', 'Axis', [q[0], N.z])   # Yaw intermediate frame
B = A.orientnew('B', 'Axis', [q[1], A.x])   # Lean intermediate frame
C = B.orientnew('C', 'Axis', [q[2], B.y])   # Disc fixed frame

# Inertial angular velocity and angular acceleration of disc fixed frame
C.set_ang_vel(N, u[0]*C.x + u[1]*C.y + u[2]*C.z)
C.set_ang_acc(N, ud[0]*C.x + ud[1]*C.y + ud[2]*C.z)

# Points
NO = Point('NO')
O = NO.locatenew('O', q[3]*N.x + q[4]*N.y + q[5]*N.z)  # Disc center
P = O.locatenew('P', r*B.z)                            # Ground contact
P.set_vel(N, 0)
camO = P.locatenew('camO', d*cam_p.x)

# Configuration constraint and its Jacobian w.r.t. q        (Table 1)
f_c = Matrix([q[5] - dot(O.pos_from(P), N.z)])
f_c_dq = f_c.jacobian(q)

# Velocity and acceleration of the center of the disc
O.set_vel(N, u[3]*C.x + u[4]*C.y + u[5]*C.z)
O.set_acc(N, O.vel(N).diff(t, C) + cross(C.ang_vel_in(N), O.vel(N)))

# Velocity level constraints                                (Table 1)
v_contact_point = O.vel(N) + cross(C.ang_vel_in(N), P.pos_from(O))
f_v = Matrix([dot(v_contact_point, uv) for uv in C])
f_v_dq = f_v.jacobian(q)
f_v_du = f_v.jacobian(u)

# Acceleration level constraints                            (Table 1)
f_a = f_v.diff(t)

# Disc angular velocity in N expressed using time derivatives of coordinates
w_c_n_qd = qd[0]*A.z + qd[1]*B.x + qd[2]*C.y
# Disc center velocity in N expressed using time derivatives of coordinates
v_o_n_qd = qd[3]*N.x + qd[4]*N.y + qd[5]*N.z

# Kinematic differential equations
kindiffs = Matrix([dot(w_c_n_qd - C.ang_vel_in(N), uv) for uv in C] +
                  [dot(v_o_n_qd - O.vel(N), uv) for uv in C])

# f_0 and f_1                                               (Table 1)
f_0 = kindiffs.subs(u_zero)
f_1 = kindiffs.subs(qd_zero)

# f_0 == f_0_coef_matrix * qd, used to solve for qdots
f_0_coef_matrix = zeros((6,6))
for i in range(6):
    for j in range(6):
        f_0_coef_matrix[i, j] = f_0[i].diff(qd[j])

# Kane's dynamic equations via elbow grease
# Partial angular velocities and velocities
partial_w_C = [C.ang_vel_in(N).diff(ui, N) for ui in u]
partial_v_O = [O.vel(N).diff(ui, N) for ui in u]

# Active forces
F_O = m*g*A.z
# Generalized active forces (unconstrained)
gaf = [dot(F_O, pv) for pv in partial_v_O]

# Inertia force
R_star_O = -m*O.acc(N)

JJ = m * r**2 / 2
II = m * r**2 / 4

# Inertia torque
I_C_O = inertia(C, II, JJ, II)     # Inertia of disc C about point O
T_star_C = -dot(I_C_O, C.ang_acc_in(N)) - cross(C.ang_vel_in(N), dot(I_C_O,
    C.ang_vel_in(N)))

# Generalized inertia forces (unconstrained)
gif = [dot(R_star_O, pv) + dot(T_star_C, pav) for pv, pav in
        zip(partial_v_O, partial_w_C)]

# Constrained dynamic equations

# Coordinates to be independent: q0, q1, q2, q3, q4
# Coordinates to be dependent: q5
# Already in the correct order, so permutation matrix is simply a 6x6 identity
# matrix
Pq = eye(6)
Pqi = Pq[:, :-1]
Pqd = Pq[:, -1]

# Speeds to be independent:  u0, u1, u2
# Speeds to be dependent:  u3, u4, u5
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
