import sys
sys.path.append(r'C:\Users\saullo\programming\abaqus')
from abaqus import mdb

import abaquslib.unstiffened_panel as up

R = 200.
H = 400.
thick = 1.
thetadeg = 60
fil_name = 'panel2_ss_lbmi_000_LB'
up.create_model_SPLA(fil_name, thetadeg, 400, 200, thick, 0., 71000,
        0.33, linear_buckling=True)
job = mdb.jobs[fil_name]
job.writeInput()
for mode_xi in [0.1, 0.2, 0.3, 0.4, 0.5, 1.]:
    modname = 'panel2_ss_lbmi_{0:03d}'.format(int(mode_xi*10))
    up.create_model_SPLA(modname, thetadeg, 400, 200, thick, 0., 71000,
            0.33, axial_displ=1., damping=1.e-7, fil_name=fil_name,
            mode_xi=mode_xi, mode_num=1)
    job = mdb.jobs[modname]
    job.writeInput()
    #job.submit()
    #job.waitForCompletion()
