

class parameters:
    def __init__(self):

        self.fsm_stand = 1
        self.fsm_stance = 2
        self.fsm_swing = 3
        self.t_stand = 0.1
        self.t_step = 0.15

        self.lz0 = -0.24864398730826576;
        self.hcl = 0.075

        self.mass = 12.453
        self.gravity = 9.81

        self.vx_min = -2.0
        self.vx_max = 2.0
        self.dvx = 0.1
        self.vy_min = -1.0
        self.vy_max = 1.0
        self.dvy = 0.05
        self.omega_min = -2;
        self.omega_max = 2;
        self.domega = 0.1;


parms = parameters();