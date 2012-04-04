from sympy import *
from sympy.physics.mechanics import *

psi, phi, theta, ox, oy, oz, w_x, w_y, w_z, v_x, v_y, v_z = dynamicsymbols('psi,'
    'phi, theta, ox, oy, oz, w_x, w_y, w_z, v_x, v_y, v_z')
psid, phid, thetad, oxd, oyd, ozd, w_xd, w_yd, w_zd, v_xd, v_yd, v_zd = dynamicsymbols('psi,'
    'phi, theta, ox, oy, oz, w_x, w_y, w_z, v_x, v_y, v_z', 1)
r, m, g, I, J = symbols('r, m, g, I, J')
t = symbols('t')

N = ReferenceFrame('N')
Y = N.orientnew('Y', 'Axis', [psi, N.z])
L = Y.orientnew('L', 'Axis', [phi, Y.x])
R = L.orientnew('R', 'Axis', [theta, L.y])

O = Point('O')
C = O.locatenew('C', r * L.z)

R.set_ang_vel(N, w_x * L.x + w_y * L.y + w_z * L.z)
O.set_vel(N, v_x * L.x + v_y * L.y + v_z * L.z)
C.v2pt_theory(O, N, R)

ccons = [r * (L.z & Y.z) - oz]
vcons = [C.vel(N) & Y.x, C.vel(N) & Y.y, C.vel(N) & Y.z]

Disk = RigidBody('Disk', O, R, m, (inertia(L, I, I, J), O))

v_o_exp_n = O.vel(N).express(N)

kd = ([psid - w_z / cos(psid), phid - w_x, thetad - w_y + w_z * tan(phi),
       (v_o_exp_n & N.x) - oxd, (v_o_exp_n & N.y) - oyd, (v_o_exp_n & N.z) - ozd])

Bodies = [Disk]
Forces = [(O, m * g * Y.z)]

KM = Kane(N)
KM.coords([psi, phi, theta, ox, oy], [oz], ccons)
KM.speeds([w_x, w_y, w_z], [v_x, v_y, v_z], vcons)
KM.kindiffeq(kd)
KM.kanes_equations(Forces, Bodies)

mm = KM.mass_matrix_full
f = KM.forcing_full
f.simplify()
f = f.subs(KM.kindiffdict())
f.simplify()

q = Matrix(KM._q)
qd = Matrix(KM._qdot)
u = Matrix(KM._u)
ud = Matrix(KM._udot)

n = len(KM._q)
l = len(KM._qdep)
o = len(KM._u)
m = len(KM._udep)

udzero = dict(zip(ud, [0] * o))
states = Matrix([KM._q + KM._u]).T
statesd = states.diff(t)

f_c = Matrix(ccons)
f_v = Matrix(vcons)
f_a = mm[n + o - m:, :] * statesd - f[n + o - m:, :]
f_0 = mm[0:n, :] * statesd
f_1 = -f[0:n, :]
f_2 = mm[n:n + o - m, :] * statesd
f_3 = -f[n:n + o - m, :]

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

#not reusable
P_q = eye(6)
P_u = eye(6)
P_qi = P_q[:, 0:5]
P_qd = P_q[:, 5:6]
P_ui = P_u[:, 0:3]
P_ud = P_u[:, 3:6]
#end not reusable

f_c_jac_q = f_c.jacobian(q)
f_v_jac_q = f_v.jacobian(q)
f_v_jac_u = f_v.jacobian(u)
C_0 = (eye(n) - P_qd * (f_c_jac_q * P_qd).inv() * f_c_jac_q) * P_qi
C_1 = -P_ud * (f_v_jac_u * P_ud).inv() * f_v_jac_q
C_2 = (eye(o) - P_ud * (f_v_jac_u * P_ud).inv() * f_v_jac_u) * P_ui

row1 = M_qq.row_join(zeros(n, o))
row2 = M_uqc.row_join(M_uuc)
row3 = M_uqd.row_join(M_uud)
M = row1.col_join(row2).col_join(row3)
#not reusable
M.simplify()

row1 = ((A_qq + A_qu * C_1) * C_0).row_join(A_qu * C_2)
row2 = ((A_uqc + A_uuc * C_1) * C_0).row_join(A_uuc * C_2)
row3 = ((A_uqd + A_uud * C_1) * C_0).row_join(A_uud * C_2)
A = row1.col_join(row2).col_join(row3)
#not reusable
A.simplify()
A = A.applyfunc(lambda x: trigsimp(x.expand(), deep=True, recursive=True))


