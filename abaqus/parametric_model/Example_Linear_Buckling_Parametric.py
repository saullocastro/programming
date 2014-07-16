###################################################################
# Parametric Study to find buckling load for different angle ply's#
# Parameters used in the parametric study:                        #
#    Theta (in degrees)                                           #
###################################################################

# create the study
BucklingStudy = ParStudy(par=('Theta'),
		     directory=ON, verbose=ON)
# define the parameters
BucklingStudy.define(DISCRETE, par='Theta', domain=(0., 10., 20., 30., 40., 50., 60., 70., 80., 90.))
#BucklingStudy.define(CONTINUOUS, par='Theta', domain=(0., 90.))
# sample the parameters
BucklingStudy.sample(INTERVAL, par=('Theta'), interval=1)
# combine the samples to give the designs
BucklingStudy.combine(MESH)
# generate analysis data
BucklingStudy.generate(template='Example_Linear_Buckling_Parametric')
# execute all analysis jobs sequentially
BucklingStudy.execute(ALL)
BucklingStudy.gather (results='bucklingload', variable='MODAL', mode=1,  step=1)
BucklingStudy.report(FILE, results='bucklingload',par='Theta', truncation=OFF, file='output.txt')


