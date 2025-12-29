import globals
import numpy as np

def inverse_kinematics_analytic(X_ref):

    L = 0.2; #thigh and shank length

    lx = X_ref[0];
    ly = X_ref[1];
    lz = X_ref[2];

    l = np.sqrt(lx**2+ly**2+lz**2);

    #sol1
    # alpha = -np.arcsin(lx/l)
    # beta = np.arcsin(ly/l);
    # gamma = np.arccos( (2*L**2-l**2)/ (2*L**2) )
    #
    # q_abduction = beta;
    # q_hip = alpha+0.5*(np.pi-gamma)
    # q_knee = -np.pi+gamma

    #sol2
    q_abduction = np.arcsin(ly/l)
    q_knee = -np.pi+np.arccos( (2*L**2-l**2)/ (2*L**2) )
    q_hip = -0.5*q_knee + np.arcsin(-lx/l)

    q_leg = np.array([q_abduction,q_hip,q_knee])

    return q_leg
