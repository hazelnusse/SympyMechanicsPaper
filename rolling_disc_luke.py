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
qd_sym = dict(zip(qd, symbols('qd1:7')))
qd_sym_inv = {v: k for (k, v) in qd_sym.items()}

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
# Disc angular velocity in N expressed using time derivatives of coordinates
w_c_n_qd = C.ang_vel_in(N)

# Inertial angular velocity and angular acceleration of disc fixed frame
C.set_ang_vel(N, u1*C.x + u2*C.y + u3*C.z)
C.set_ang_acc(N, u1d*C.x + u2d*C.y + u3d*C.z)
print("Angular acceleration")
mprint(C.ang_acc_in(N))

# Points
NO = Point('NO')                                       # Inertial origin
CO = NO.locatenew('CO', q4*N.x + q5*N.y + q6*N.z)      # Disc center
# Disc center velocity in N expressed using time derivatives of coordinates
v_co_n_qd = CO.pos_from(NO).dt(N)

CO.set_vel(N, u4*C.x + u5*C.y + u6*C.z)
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
f_0_coef_matrix = f_0.jacobian(qd)

# Acceleration level constraints                            (Table 1)
#v_co_n = cross(C.ang_vel_in(N), CO.pos_from(P))
#a_co_n = v_co_n.dt(B) + cross(B.ang_vel_in(N), v_co_n)
#f_a = Matrix([((a_co_n - CO.acc(N)) & uv).expand() for uv in B])
f_a = f_v.diff(t)
print("Acceleration level constraints")
mprint(f_a)

# Kane's dynamic equations via elbow grease
# Partial angular velocities and velocities
partial_w_C = [C.ang_vel_in(N).diff(ui, N) for ui in u]
partial_v_CO = [CO.vel(N).diff(ui, N) for ui in u]

# Active forces
F_CO = m*g*A.z
# Generalized active forces (unconstrained)
gaf = [dot(F_CO, pv) for pv in partial_v_CO]
print("Generalized active forces (unconstrained)")
mprint(gaf)

# Inertia force
R_star_CO = -m*CO.acc(N)

I = (m * r**2) / 4
J = (m * r**2) / 2

# Inertia torque
I_C_CO = inertia(C, I, J, I)     # Inertia of disc C about point CO
T_star_C = -(dot(I_C_CO, C.ang_acc_in(N))
             + cross(C.ang_vel_in(N), dot(I_C_CO, C.ang_vel_in(N))))

# Generalized inertia forces (unconstrained)
gif = [dot(R_star_CO, pv) + dot(T_star_C, pav) for pv, pav in
        zip(partial_v_CO, partial_w_C)]
print("Generalized inertia forces (unconstrained)")

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

print("Generalized active forces (constrained)")
mprint(gaf_con)

# Build the part of f_2 and f_3 that come from Kane's equations, the first three
# rows of each
f_2 = gif_con.subs(ud_sym).subs(u_zero).subs(qd_zero).subs(ud_sym_inv)
f_3 = gif_con.subs(ud_zero) + gaf_con

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

#qdep = Pqd.T * Matrix(q)
#udep = Pud.T * Matrix(u)

#n = len(q)
#l = len(qdep)
#o = len(u)
#m = len(udep)

# Point of linearization
# equilibrium conditions:
# Non-zero variables: q2, q1d, q3d, q4d, q5d
eq_q = {q1: 0,          # yaw angle (ignorable)
        q3: 0,          # spin angle (ignorable)
        q4: 0,          # x of disc center (ignorable)
        q5: 0,          # y of disc center (ignorable)
        q6: -r*cos(q2)} # z of disc center
eq_qd = {q2d: 0}        # lean rate

# Dependent generalized speeds in terms of independent ones
eq_u = solve(f_v.subs(eq_q), [u4, u5, u6])

# Solve kinematic equations for qdots in terms of independent u's
kindiffs_eq = kindiffs.subs(eq_u).subs(eq_qd).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)
soln = solve(kindiffs_eq, [u1, u2, u3, q4d, q5d, q6d])

for ui in [u1, u2, u3]:
    eq_u[ui] = soln[ui]

for ui in [u4, u5, u6]:
    eq_u[ui] = eq_u[ui].subs(soln)

for qdi in [q4d, q5d, q6d]:
    eq_qd[qdi] = soln[qdi]

print("Equilibrium coordinates:")
for qi in q:
    if qi in eq_q.keys():
        s = msprint(qi) + " = " + msprint(eq_q[qi])
        print(s)

print("Equilibrium coordinate time derivatives:")
for qdi in qd:
    if qdi in eq_qd.keys():
        s = msprint(qdi) + " = " + msprint(eq_qd[qdi])
        print(s)

print("Equilibrium generalized speeds:")
for ui in u:
    if ui in eq_u.keys():
        s = msprint(ui) + " = " + msprint(eq_u[ui])
        print(s)

# Solve differentiated acceleration constraints for dependent du/dt's
f_a_eq = f_a.subs(eq_qd).subs(ud_sym).subs(eq_u).subs(qd_sym).subs(eq_q).subs(qd_sym_inv).subs(ud_sym_inv)
eq_ud = solve(f_a_eq, [u4d, u5d, u6d])

# Evaluate the dynamic equations using the dependent du/dt's
dyndiffs_eq = (f_2 + f_3).subs(eq_ud).subs(eq_qd).subs(ud_sym).subs(eq_u).subs(qd_sym).subs(eq_q).subs(qd_sym_inv).subs(ud_sym_inv).expand()

mprint(dyndiffs_eq)
# Solve the dynamic equations for the remaining two independent du/dt's
soln = solve(dyndiffs_eq, [u1d, u2d, u3d])

# Update the equilibrium dependent du/dt's with solution for the independent du/dt's
for udi in [u4d, u5d, u6d]:
    eq_ud[udi] = eq_ud[udi].subs(soln)

for udi in [u1d, u2d, u3d]:
    eq_ud[udi] = soln[udi]

print("Equilibrium generalized speed time derivatives:")
for udi in ud:
    if udi in eq_ud.keys():
        s = msprint(udi) + " = " + msprint(eq_ud[udi])
        print(s)

print("f_c, f_v, f_a, f_0 + f_1, f_2 + f_3, evaluated at equilibrium conditions")
f_c_eq = f_c.subs(eq_q)
f_v_eq = f_v.subs(eq_u).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)
f_a_eq = f_a.subs(eq_ud).subs(eq_u).subs(qd_sym).subs(eq_q).subs(qd_sym_inv).expand()
f_0_eq_plus_f_1_eq = (f_0 + f_1).subs(eq_qd).subs(eq_u).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)
f_2_eq_plus_f_3_eq = (f_2 + f_3).subs(eq_ud).subs(eq_u).subs(qd_sym).subs(eq_q).subs(qd_sym_inv).expand()
f_2_eq_plus_f_3_eq.simplify()
mprint(f_c_eq)
mprint(f_v_eq)
mprint(f_a_eq)
mprint(f_0_eq_plus_f_1_eq)
mprint(f_2_eq_plus_f_3_eq)

"""
eom_eq = (gif_con + gaf_con).subs(ud_sym).subs(eq_qd).subs(eq_u).subs(qd_sym).subs(eq_q).subs(qd_sym_inv).subs(ud_sym_inv).expand()
mprint(eom_eq)
p = Poly(steady_turn_eqn, q1d)
a, b, c = p.coeffs()
discriminant = b*b - 4*a*c
print("Steady turning balance equation:")
print(msprint(steady_turn_eqn) + " = 0")
print("Coefficients of quadratic in dq1/dt:")
print("a = " + msprint(a))
print("b = " + msprint(b))
print("c = " + msprint(c))
print("Discriminant which must be non-negative:")
mprint(discriminant.simplify())
stop
"""

M_qq = f_0.jacobian(qd).subs(eq_ud).subs(eq_u).subs(eq_qd).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)
M_uqc = f_a.jacobian(qd).subs(eq_ud).subs(eq_u).subs(eq_qd).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)
M_uuc = f_a.jacobian(ud).subs(eq_ud).subs(eq_u).subs(eq_qd).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)
M_uqd = f_2.jacobian(qd).subs(eq_ud).subs(eq_u).subs(eq_qd).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)
M_uud = f_2.jacobian(ud).subs(eq_ud).subs(eq_u).subs(eq_qd).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)

A_qq = -(f_0 + f_1).jacobian(q).subs(eq_ud).subs(eq_u).subs(eq_qd).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)

A_qu = -f_1.jacobian(u).subs(eq_ud).subs(eq_u).subs(eq_qd).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)
A_uqc = -f_a.jacobian(q).subs(eq_ud).subs(eq_u).subs(eq_qd).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)
A_uuc = -f_a.jacobian(u).subs(eq_ud).subs(eq_u).subs(eq_qd).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)
A_uqd = -(f_2 + f_3).jacobian(q).subs(eq_ud).subs(eq_u).subs(eq_qd).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)
A_uud = -f_3.jacobian(u).subs(eq_ud).subs(eq_u).subs(eq_qd).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)

f_c_jac_q = f_c.jacobian(q).subs(eq_ud).subs(eq_u).subs(eq_qd).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)
f_v_jac_q = f_v.jacobian(q).subs(eq_ud).subs(eq_u).subs(eq_qd).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)
f_v_jac_u = f_v.jacobian(u).subs(eq_ud).subs(eq_u).subs(eq_qd).subs(qd_sym).subs(eq_q).subs(qd_sym_inv)

C_0 = (eye(len(q)) - Pqd * (f_c_jac_q * Pqd).inv() * f_c_jac_q) * Pqi
C_1 = -Pud * (f_v_jac_u * Pud).inv() * f_v_jac_q
C_2 = (eye(len(u)) - Pud * (f_v_jac_u * Pud).inv() * f_v_jac_u) * Pui

row1 = M_qq.row_join(zeros(len(q), len(u)))
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
print("A:")
mprint(Amat)

upright_nominal = {q1d: 0, q2: 0, m: 1, r: 1, g: 1}
M_upright = M.subs(upright_nominal)
A_upright = Amat.subs(upright_nominal)
A_ss = M_upright.inv() * A_upright
pprint(A_ss)
perm = Pqi.row_join(zeros(6, 3)).col_join(zeros(6, 5).row_join(Pui))
A_ss_red = perm.T * A_ss
pprint(A_ss_red)

evals = A_ss_red.eigenvals()
print(evals.keys())
q3d_check = 1/sqrt(3)
print([evalf.N(i.subs({q3d: q3d_check})) for i in evals.keys()])

