import numpy as np
from parameters import parms

def init():
    global time
    global fsm,t_fsm
    
    time = 0
    fsm = np.array([parms.fsm_stand,parms.fsm_stand,parms.fsm_stand,parms.fsm_stand])
    t_fsm = np.zeros(4)

    #global lz0
    #global hcl

    
    

    #cart traj
    global t_i, t_f
    global lx_ref, ly_ref, lz_ref, lxdot_ref, lydot_ref, lzdot_ref
    global lx_i, lx_f, ly_i, ly_f, lz_i, lz_f

    t_i = np.zeros(4);
    t_f = np.array([parms.t_stand,parms.t_stand,parms.t_stand,parms.t_stand])
    lx_ref, ly_ref, lz_ref, lxdot_ref, lydot_ref, lzdot_ref = (np.zeros(4) for _ in range(6))
    lx_i, ly_i, lx_f, ly_f = (np.zeros(4) for _ in range(4))
    lz_i = np.array([parms.lz0,parms.lz0,parms.lz0,parms.lz0]);
    lz_f = np.array([parms.lz0,parms.lz0,parms.lz0,parms.lz0]);

    #joint traj
    global q_ref, u_ref

    q_ref,u_ref = (np.zeros(12) for _ in range(2))
    
    #joint control
    global q_act, u_act, trq
    

    q_act,u_act,trq = (np.zeros(12) for _ in range(3))

    global xdot_ref, ydot_ref, psidot_ref

    xdot_ref = 0
    ydot_ref = 0
    psidot_ref = 0

    global prev_step, step

    step = 0
    prev_step = 0

    global pos_quat_trunk, vel_angvel_trunk

    pos_quat_trunk = np.zeros(7) #[x,y,z,q0,qx,qy,qz]
    vel_angvel_trunk = np.zeros(6) #[vx,vy,vz,omegab_x,omegab_y,omegab_z] omegab is body frame
