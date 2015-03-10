import sys
sys.path.append(r'C:\Users\saullo\programming\abaqus')
from abaqus import mdb

import abaquslib.unstiffened_panel as up

R = 200.
H = 400.
thick = 1.
thetadeg = 60
modname = 'panel2_ss_LB'
up.create_model_SPLA(modname, thetadeg, 400, 200, thick, 0., 71000,
        0.33, linear_buckling=True)
mdb.jobs[modname].writeInput()
for load in [0.01,  1, 5, 10, 20, 30, 40, 50, 70]:
    modname = 'panel2_ss_PL_{0:03d}'.format(int(load))
    up.create_model_SPLA(modname, thetadeg, 400, 200, thick, load, 71000, 0.33,
            axial_displ=1.5, damping=1.e-7)
    job = mdb.jobs[modname]
    job.writeInput()
    #job.submit()
    #job.waitForCompletion()
