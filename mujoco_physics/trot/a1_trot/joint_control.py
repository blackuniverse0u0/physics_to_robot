import globals
import numpy as np
from parameters import parms
from jac_end_effector_leg import jac_end_effector_leg
from stance_force import stance_force

def joint_control():

    fsm = globals.fsm
    fsm_stance = parms.fsm_stance
    fsm_swing = parms.fsm_swing
    fsm_stand = parms.fsm_stand

    q_act = globals.q_act
    u_act = globals.u_act
    q_ref = globals.q_ref
    u_ref = globals.u_ref

    for leg_no in range(4):
        if (fsm[leg_no]==fsm_stand):
            #simple pd control: trq = -kp*(q_act-q_ref)-kd*(u_act-u_ref)
            gain = 10
            globals.trq[3*leg_no] =  gain*(-10*(q_act[3*leg_no] -q_ref[3*leg_no] )-1*(u_act[3*leg_no] -u_ref[3*leg_no] ))
            globals.trq[3*leg_no+1] =  gain*(-10*(q_act[3*leg_no+1] -q_ref[3*leg_no+1] )-1*(u_act[3*leg_no+1] -u_ref[3*leg_no+1] ))
            globals.trq[3*leg_no+2] =  gain*(-10*(q_act[3*leg_no+2] -q_ref[3*leg_no+2] )-1*(u_act[3*leg_no+2] -u_ref[3*leg_no+2] ))

        if (fsm[leg_no]==fsm_stance):
            #simple pd control: trq = -kp*(q_act-q_ref)-kd*(u_act-u_ref)
            #F = np.array([0,0,0.5*parms.mass*parms.gravity])
            if (leg_no==0 or leg_no==1):
                F0,F1,F2,F3 = stance_force(leg_no)

            if (leg_no==0):
                F = F0
            if (leg_no==1):
                F = F1
            if (leg_no==2):
                F = F2
            if (leg_no==3):
                F = F3


            q_leg = np.array([q_ref[3*leg_no],q_ref[3*leg_no+1],q_ref[3*leg_no+2]])
            J = jac_end_effector_leg(q_leg,leg_no)
            trq_grav = -J.T@F
            #trq_grav = np.zeros(3)
            gain = 10
            globals.trq[3*leg_no] =  trq_grav[0] + gain*(-10*(q_act[3*leg_no] -q_ref[3*leg_no] )-1*(u_act[3*leg_no] -u_ref[3*leg_no] ))
            globals.trq[3*leg_no+1] =  trq_grav[1] + gain*(-10*(q_act[3*leg_no+1] -q_ref[3*leg_no+1] )-1*(u_act[3*leg_no+1] -u_ref[3*leg_no+1] ))
            globals.trq[3*leg_no+2] =  trq_grav[2] + gain*(-10*(q_act[3*leg_no+2] -q_ref[3*leg_no+2] )-1*(u_act[3*leg_no+2] -u_ref[3*leg_no+2] ))

        if (fsm[leg_no]==fsm_swing):
            #simple pd control: trq = -kp*(q_act-q_ref)-kd*(u_act-u_ref)
            gain = 10
            globals.trq[3*leg_no] =  gain*(-10*(q_act[3*leg_no] -q_ref[3*leg_no] )-1*(u_act[3*leg_no] -u_ref[3*leg_no] ))
            globals.trq[3*leg_no+1] =  gain*(-10*(q_act[3*leg_no+1] -q_ref[3*leg_no+1] )-1*(u_act[3*leg_no+1] -u_ref[3*leg_no+1] ))
            globals.trq[3*leg_no+2] =  gain*(-10*(q_act[3*leg_no+2] -q_ref[3*leg_no+2] )-1*(u_act[3*leg_no+2] -u_ref[3*leg_no+2] ))
