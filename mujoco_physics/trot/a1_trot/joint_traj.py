import globals
import numpy as np
from parameters import parms
from inverse_kinematics_analytic import inverse_kinematics_analytic
from jac_end_effector_leg import jac_end_effector_leg

def joint_traj():

    lx_ref = globals.lx_ref;
    ly_ref = globals.ly_ref;
    lz_ref = globals.lz_ref;
    lxdot_ref = globals.lxdot_ref;
    lydot_ref = globals.lydot_ref;
    lzdot_ref = globals.lzdot_ref;


    for leg_no in range(4):

        X_ref = np.array([lx_ref[leg_no],ly_ref[leg_no],lz_ref[leg_no]])
        Xdot_ref = np.array([lxdot_ref[leg_no], lydot_ref[leg_no], lzdot_ref[leg_no]])
        q_leg = inverse_kinematics_analytic(X_ref)
        globals.q_ref[3*leg_no:3*leg_no+3] = q_leg.copy();

        J_foot = jac_end_effector_leg(q_leg,leg_no)
        J_foot_inv = np.linalg.inv(J_foot)
        u_leg = J_foot_inv@Xdot_ref
        globals.u_ref[3*leg_no:3*leg_no+3] = u_leg.copy();
