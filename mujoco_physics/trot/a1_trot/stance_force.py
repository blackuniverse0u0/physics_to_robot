import globals
import numpy as np
from parameters import parms
from forward_kinematics_robot import forward_kinematics_robot
import utility as ram


def stance_force(leg_no):

    q_act = globals.q_act.copy()
    pos_quat_trunk = globals.pos_quat_trunk.copy()
    vel_angvel_trunk = globals.vel_angvel_trunk.copy()

    quat = pos_quat_trunk[3:]
    euler = ram.quat2bryant(quat)

    Rz = ram.rotation(euler[2],2)
    R = ram.quat2mat(quat)
    R_body = Rz.T@R
    quat_body = ram.mat2quat(R_body)
    pos_quat_trunk_ = np.concatenate((pos_quat_trunk[:3],quat_body))
    vel = vel_angvel_trunk[:3]
    vel_body = Rz.T@vel

    q = np.concatenate((pos_quat_trunk_,q_act))

    _,sol = forward_kinematics_robot(q)

    #set up A matrix (AF = b)
    _, sol = forward_kinematics_robot(q)
    end_eff_pos = sol.end_eff_pos
    trunk_com_pos = sol.trunk_com_pos

    I = np.identity(3)
    if (leg_no == 0):
        r0=end_eff_pos[0,:]-trunk_com_pos;
        r3=end_eff_pos[3,:]-trunk_com_pos;
        R0 = ram.vec2skew(r0)
        R3 = ram.vec2skew(r3)
        A = np.block([
            [I, I],     # Top row
            [R0, R3]    # Bottom row
        ])

    if (leg_no == 1):
        r1=end_eff_pos[1,:]-trunk_com_pos;
        r2=end_eff_pos[2,:]-trunk_com_pos;
        R1 = ram.vec2skew(r1)
        R2 = ram.vec2skew(r2)
        A = np.block([
            [I, I],     # Top row
            [R1, R2]    # Bottom row
        ])

    z = pos_quat_trunk[2]
    z_ref = -parms.lz0
    zdot = vel_angvel_trunk[2]
    omega = vel_angvel_trunk[3:]
    xdot = vel_body[0]
    ydot = vel_body[1]
    psidot = vel_angvel_trunk[5]

    #b matrix
    fx0 = 100*(globals.xdot_ref-xdot)
    fy0 = 100*(globals.ydot_ref-ydot)
    fz0 = 50*(-10*(z-z_ref)-1*zdot) + parms.mass*parms.gravity
    Mx0 = 50*(-10*euler[0]-0.5*omega[0])
    My0 = 50*(-10*euler[1]-0.5*omega[1])
    Mz0 = 10*(globals.psidot_ref-psidot)

    b = np.array([fx0,fy0,fz0,Mx0,My0,Mz0])
    #F = np.array([0,0,0.5*parms.mass*parms.gravity])

      #compute F = inv(A)*b
    Ainv = np.linalg.pinv(A,rcond=1e-10,hermitian=False)
    F = Ainv.dot(b)

    if (leg_no == 0):
        F0 = np.array([F[0],F[1],F[2]])
        F3 = np.array([F[3],F[4],F[5]])
        F1 = np.zeros(3)
        F2 = np.zeros(3)

    if (leg_no == 1):
        F1 = np.array([F[0],F[1],F[2]])
        F2 = np.array([F[3],F[4],F[5]])
        F0 = np.zeros(3)
        F3 = np.zeros(3)

    return F0,F1,F2,F3

