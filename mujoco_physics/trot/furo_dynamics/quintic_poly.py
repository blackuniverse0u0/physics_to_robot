import numpy as np

def quintic_poly(t,t0,tf,q0,qf):

    A = np.array([ [1, t0, t0**2, t0**3, t0**4, t0**5],
                    [1, tf, tf**2, tf**3, tf**4, tf**5],
                    [0,  1, 2*t0,  3*t0**2, 4*t0**3, 5*t0**4 ],
                    [0,  1, 2*tf,  3*tf**2, 4*tf**3, 5*tf**4],
                    [0,  0, 2,  6*t0, 12*t0**2, 20*t0**3 ],
                    [0,  0, 2,  6*tf, 12*tf**2, 20*tf**3],
                     ])
    b = np.array([ [q0],
                    [qf],
                    [0],
                    [0],
                    [0],
                    [0]
                     ])

    A_inv = np.linalg.inv(A)
    a = A_inv@b

    if (t<t0):
        t = t0;

    if (t>tf):
        t=tf

    q = a[0]+a[1]*t+a[2]*t**2+a[3]*t**3+a[4]*t**4+a[5]*t**5;
    qdot = a[1]+2*a[2]*t+3*a[3]*t**2+4*a[4]*t**3+5*a[5]*t**4;
    qddot = 2*a[2]+6*a[3]*t+12*a[4]*t**2+20*a[5]*t**3;

    return q,qdot,qddot
