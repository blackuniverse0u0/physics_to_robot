import globals
from parameters import parms

def state_machine():

    time = globals.time
    #print(globals.time)
    #print(globals.fsm)

    fsm_stand = parms.fsm_stand
    fsm_stance = parms.fsm_stance
    fsm_swing = parms.fsm_swing

    t_stand = parms.t_stand
    t_step = parms.t_step

    c = 0.183 #from xml

    for leg_no in range(4):
        if (time>=globals.t_fsm[leg_no]+t_stand and globals.fsm[leg_no]==fsm_stand):

            if (leg_no==0 or leg_no==3):
                globals.fsm[leg_no]= fsm_swing
                globals.t_fsm[leg_no] = time;
                globals.t_i[leg_no] = 0;
                globals.t_f[leg_no] = t_step
                globals.lz_i[leg_no] = parms.lz0;
                globals.lz_f[leg_no] = parms.lz0+parms.hcl;

            if (leg_no==1 or leg_no==2):
                globals.fsm[leg_no]= fsm_stance
                globals.t_fsm[leg_no] = time;
                globals.t_i[leg_no] = 0;
                globals.t_f[leg_no] = t_step
                globals.lz_i[leg_no] = parms.lz0;
                globals.lz_f[leg_no] = parms.lz0;

        if (time>=globals.t_fsm[leg_no]+t_step and globals.fsm[leg_no]==fsm_stance):
            globals.fsm[leg_no]= fsm_swing
            globals.t_fsm[leg_no] = time;
            globals.t_i[leg_no] = 0;
            globals.t_f[leg_no] = t_step
            globals.lz_i[leg_no] = parms.lz0;
            globals.lz_f[leg_no] = parms.lz0+parms.hcl;
        
            globals.lx_i[leg_no] = -0.5*globals.xdot_ref*t_step
            globals.lx_f[leg_no] = 0.5*globals.xdot_ref*t_step
            globals.ly_i[leg_no] = -0.5*globals.ydot_ref*t_step
            globals.ly_f[leg_no] = 0.5*globals.ydot_ref*t_step

            if (leg_no==0 or leg_no==1): #front legs
                globals.ly_i[leg_no] -= 0.5*c*globals.psidot_ref*t_step
                globals.ly_f[leg_no] += 0.5*c*globals.psidot_ref*t_step
            else:
                globals.ly_i[leg_no] += 0.5*c*globals.psidot_ref*t_step
                globals.ly_f[leg_no] -= 0.5*c*globals.psidot_ref*t_step



        if (time>=globals.t_fsm[leg_no]+t_step and globals.fsm[leg_no]==fsm_swing):
            if (leg_no == 0 or leg_no==1):
                globals.step +=1
            globals.fsm[leg_no]= fsm_stance
            globals.t_fsm[leg_no] = time;
            globals.t_i[leg_no] = 0;
            globals.t_f[leg_no] = t_step
            globals.lz_i[leg_no] = parms.lz0;
            globals.lz_f[leg_no] = parms.lz0;

            globals.lx_i[leg_no] = 0.5*globals.xdot_ref*t_step
            globals.lx_f[leg_no] = -0.5*globals.xdot_ref*t_step
            globals.ly_i[leg_no] = 0.5*globals.ydot_ref*t_step
            globals.ly_f[leg_no] = -0.5*globals.ydot_ref*t_step

            if (leg_no==0 or leg_no==1): #front legs
                globals.ly_i[leg_no] += 0.5*c*globals.psidot_ref*t_step
                globals.ly_f[leg_no] -= 0.5*c*globals.psidot_ref*t_step
            else:
                globals.ly_i[leg_no] -= 0.5*c*globals.psidot_ref*t_step
                globals.ly_f[leg_no] += 0.5*c*globals.psidot_ref*t_step



