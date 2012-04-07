from sympy import symbols, Matrix
from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point, dot,
cross, mprint, RigidBody, inertia, Kane)

# Symbols for time and constant parameters
t, r, m, g, I, J = symbols('t r m g I J')

# Configuration variables and their time derivatives
# q[0] -- yaw
# q[1] -- lean
# q[2] -- spin
# q[3] -- disc center distance from inertial origin, x direction
# q[4] -- disc center distance from inertial origin, y direction
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
N = ReferenceFrame('N')                     # Inertial reference frame
A = N.orientnew('A', 'Axis', [q[0], N.z])   # Yaw intermediate frame
B = A.orientnew('B', 'Axis', [q[1], A.x])   # Lean intermediate frame
C = B.orientnew('C', 'Axis', [q[2], B.y])   # Disc fixed frame

# Inertial angular velocity and angular acceleration of disc fixed frame
C.set_ang_vel(N, u[0]*C.x + u[1]*C.y + u[2]*C.z)
C.set_ang_acc(N, ud[0]*C.x + ud[1]*C.y + ud[3]*C.z)

# Points
P = Point('P')                              # Ground disc contact point
O = P.locatenew('O', -r*B.z)                # Center of disc

# Configuration constraint and its Jacobian w.r.t. q
f_c = Matrix([q[5] - dot(O.pos_from(P), N.z)])
f_c_dq = f_c.jacobian(q)

# Velocity and acceleration of the center of the disc
O.set_vel(N, u[3]*C.x + u[4]*C.y + u[5]*C.z)
O.set_acc(N, O.vel(N).diff(t, C) + cross(C.ang_vel_in(N), O.vel(N)))

# Velocity level constraints
v_contact_point = O.vel(N) + cross(C.ang_vel_in(N), P.pos_from(O))
f_v = Matrix([dot(v_contact_point, uv) for uv in C])
f_v_dq = f_v.jacobian(q)
f_v_du = f_v.jacobian(u)

# Acceleration level constraints
f_a = f_v.diff(t)

# Disc angular velocity in N expressed using time derivatives of coordinates
w_c_n_qd = qd[0]*A.z + qd[1]*B.x + qd[2]*C.y
# Disc center velocity in N expressed using time derivatives of coordinates
v_o_n_qd = qd[3]*N.x + qd[4]*N.y + qd[5]*N.z
# Kinematic differential equations
kindiffs = Matrix([dot(w_c_n_qd - C.ang_vel_in(N), uv) for uv in C] + 
                  [dot(v_o_n_qd - O.vel(N), uv) for uv in C])

# f_0 and f_1 from Table 1
f_0 = kindiffs.subs({ui : 0 for ui in u})
f_1 = kindiffs.subs({qdi : 0 for qdi in qd})

# Kane's dynamic equations
Bodies_List = [RigidBody('Disk', O, C, m, (inertia(C, I, J, I), O))]
Forces_List = [(O, m*g*A.z)]

KM = Kane(N)
KM.coords(q[:5], [q[5]], f_c)
KM.speeds(u[:3], u[3:], f_v)
KM.kindiffeq(kindiffs)
KM.kanes_equations(Forces_List, Bodies_List)
mm = KM.mass_matrix_full
mprint(mm)
