from sympy import symbols, Matrix, eye, zeros, latex, pi
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point, dot,
cross, mprint, RigidBody, inertia, Kane, mlatex)

# Symbols for time and constant parameters
t, r, m, g, I, J = symbols('t r m g I J')

# Configuration variables and their time derivatives
# q[0] -- yaw
# q[1] -- lean
# q[2] -- spin
# q[3] -- disc contact point distance from inertial origin, x direction
# q[4] -- disc contact point distance from inertial origin, y direction
# q[5] -- disc center distance from inertial origin, z direction
q = dynamicsymbols('q:6')
qd = [qi.diff(t) for qi in q]

# Generalized speeds and their time derivatives
# u[0] -- disc angular velocity component, disc fixed x direction
# u[1] -- disc angular velocity component, disc fixed y direction
# u[2] -- disc angular velocity component, disc fixed z direction
# u[3] -- disc velocity component, disc fixed x direction
# u[4] -- disc velocity component, disc fixed y direction
# u[5] -- disc velocity component, disc fixed z direction
u = dynamicsymbols('u:6')
ud = [ui.diff(t) for ui in u]

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
P = NO.locatenew('P', q[3]*N.x + q[4]*N.y)  # Ground disc contact point
P.set_vel(N, 0)
O = P.locatenew('O', -r*B.z)                # Center of disc
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
v_o_n_qd = qd[3]*N.x + qd[4]*N.y + cross(qd[0]*A.z + qd[1]*B.x, -r*B.z)
# Kinematic differential equations
kindiffs = Matrix([dot(w_c_n_qd - C.ang_vel_in(N), uv) for uv in C] +
                  [dot(v_o_n_qd - O.vel(N), uv) for uv in C])

# f_0 and f_1                                               (Table 1)
f_0 = kindiffs.subs({ui : 0 for ui in u})
f_1 = kindiffs.subs({qdi : 0 for qdi in qd})

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

# Inertia torque
I_C_O = inertia(C, I, J, I)     # Inertia of disc C about point O
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
    f_3[i] = gif_con[i].subs({udi : 0 for udi in ud}) + gaf_con[i]

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
