"""
Unstiffened Panels (:mod:`abaqus.unstiffened_panel`)
====================================================

.. currentmodule:: abaqus.unstiffened_panel

"""
from multiprocessing import cpu_count

from math import cos, sin, radians, pi

import abaqus_functions


def get_bc(string):
    from abaqusConstants import SET, UNSET
    u1 = SET if '1' in string else UNSET
    u2 = SET if '2' in string else UNSET
    u3 = SET if '3' in string else UNSET
    ur1 = SET if '4' in string else UNSET
    ur2 = SET if '5' in string else UNSET
    ur3 = SET if '6' in string else UNSET

    return u1, u2, u3, ur1, ur2, ur3


def create_model_SPLA(mod_name, thetadeg, H, R, thickness, PL, E, nu,
        axial_load=None, axial_displ=None, damping=1.e-8, meshsize=None,
        bc_side='12', numCpus=None, linear_buckling=False, fil_name=None,
        mode_xi=None,
        mode_num=None):
    """Create a model of a simply supported unstiffened panel for the SPLA

    Parameters
    ----------
    mod_name : str
        Model name.
    thetadeg : float
        Panel circumferential angle.
    H : float
        Panel height.
    R : float
        Panel radius.
    thickness : float
        Panel thickness.
    PL : float
        Perturbation load magnitude.
    E : float
        Young modulus
    nu : float
        Poisson's ratio.
    axial_load : float or None
        Resultant of the applied axial load. If ``None`` activates displacement
        controlled axial compression.
    axial_displ : float or None
        Axial displacement.
    damping : float, optional
        The artificial damping factor.
    meshsize : int or None, optional
        The mesh size using metric units. When ``None`` is used it is estimated
        based on the panel dimensions.
    bc_side : str, optional
        The degrees-of-freedom to be constrained at the side edges.
    numCpus : int or None, optional
        The number of CPUs for the corresponding job that will be created in
        Abaqus.
    linear_buckling : bool, optional
        Flag indicating if this model should be a linear buckling model.
    fil_name : str or None, optional
        The .fil file with the imperfection pattern.
    mode_xi : float or None, optional
        The imperfection amplitude.
    mode_num : int or None, optional
        The linear buckling mode to be applied as imperfection.

    """
    from abaqusConstants import *
    from abaqus import mdb

    PL = float(abs(PL))
    if axial_load is not None:
        axial_load = float(abs(axial_load))
        load_controlled = True
    else:
        if axial_displ is None:
            axial_displ = 0.005*H
        else:
            axial_displ = float(abs(axial_displ))
        load_controlled = False

    thetarad = radians(thetadeg)
    xref = R*cos(thetarad/2.)
    yref = R*sin(thetarad/2.)

    if meshsize is None:
        meshsize = R*2*pi/420.

    mod = mdb.Model(name=mod_name, modelType=STANDARD_EXPLICIT)
    ra = mod.rootAssembly
    s = mod.ConstrainedSketch(name='__profile__', sheetSize=2*H)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.ArcByCenterEnds(center=(0.0, 0.0),
                      point1=(xref, yref),
                      point2=(xref, -yref), direction=CLOCKWISE)
    part = mod.Part(name='Part-1', dimensionality=THREE_D,
                    type=DEFORMABLE_BODY)
    datums = part.datums
    part.BaseShellExtrude(sketch=s, depth=H)
    hplane = part.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE,
            offset=H/2.)
    part.PartitionFaceByDatumPlane(datumPlane=datums[hplane.id],
            faces=part.faces)
    vplane = part.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE,
            offset=0.0)
    part.PartitionFaceByDatumPlane(datumPlane=datums[vplane.id],
            faces=part.faces)

    s.unsetPrimaryObject()
    del mod.sketches['__profile__']
    part.seedPart(size=meshsize, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()
    ra.Instance(name='Part-1-1', part=part, dependent=ON)
    ra.regenerate()
    inst = ra.instances['Part-1-1']

    csys_cyl = ra.DatumCsysByThreePoints(name='csys_cyl',
                                         coordSysType=CYLINDRICAL,
                                         origin=(0.0, 0.0, 0.0),
                                         point1=(1.0, 0.0, 0.0),
                                         point2=(0.0, 1.0, 0.0))
    csys_cyl = ra.datums[csys_cyl.id]

    vertices = inst.vertices
    reference_point = vertices.findAt(((R, 0, H),))
    ra.Set(vertices=reference_point, name='set_RP')
    es = inst.edges
    top_edges = es.getByBoundingBox(0, -2*R, 0.99*H, 2*R, 2*R, 1.01*H)
    set_top_edges = ra.Set(edges=top_edges, name='top_edges')

    if not linear_buckling:
        mod.StaticStep(name='constant_loads',
            previous='Initial', maxNumInc=100,
            stabilizationMethod=NONE, continueDampingFactors=False,
            adaptiveDampingRatio=None, initialInc=1.0, maxInc=1.0, nlgeom=ON)
        mod.StaticStep(name='variable_loads', previous='constant_loads',
                maxNumInc=10000, stabilizationMagnitude=1e-08,
                stabilizationMethod=DAMPING_FACTOR, continueDampingFactors=False,
                adaptiveDampingRatio=None, initialInc=0.01, maxInc=0.01)

        if PL > 0:
            perturbation_point = vertices.findAt(((R, 0, H/2.),))
            region = ra.Set(vertices=perturbation_point,
                            name='perturbation_point')
            mod.ConcentratedForce(name='perturbation_load',
                    createStepName='constant_loads', region=region, cf1=-PL,
                    distributionType=UNIFORM, field='', localCsys=csys_cyl)

        mod.HistoryOutputRequest(name='history_RP',
            createStepName='constant_loads', variables=('U3', 'RF3'),
            region=ra.sets['set_RP'], sectionPoints=DEFAULT, rebar=EXCLUDE)

        if load_controlled:
            load_top = axial_load/(R*thetarad)
            region = ra.Surface(side1Edges=top_edges, name='Surf-1')
            mod.ShellEdgeLoad(name='load_top', createStepName='variable_loads',
                    region=region, magnitude=load_top, distributionType=UNIFORM,
                    field='', localCsys=csys_cyl)
        else:
            mod.DisplacementBC(name='axial_displacement',
                    createStepName='variable_loads', region=set_top_edges,
                    u1=UNSET, u2=UNSET, u3=-axial_displ, ur1=UNSET, ur2=UNSET,
                    ur3=UNSET, amplitude=UNSET, fixed=OFF,
                    distributionType=UNIFORM, fieldName='', localCsys=csys_cyl)

        if all([fil_name, mode_xi, mode_num]):
            if fil_name.endswith('.fil'):
                fil_name = fil_name[:-4]
            text = '*IMPERFECTION, STEP=1, FILE={0}'.format(fil_name)
            text += '\n{0:d}, {1:f}'.format(int(mode_num), float(mode_xi))
            text += '\n**'
            pattern = '*Step'
            abaqus_functions.edit_keywords(mod=mod, text=text,
                                           before_pattern=pattern)


    else:
        mod.BuckleStep(name='linear_buckling', previous='Initial',
                numEigen=50, eigensolver=LANCZOS, minEigen=0.0,
                blockSize=DEFAULT, maxBlocks=DEFAULT)
        region = ra.Surface(side1Edges=top_edges, name='Surf-1')
        mod.ShellEdgeLoad(name='load_top', createStepName='linear_buckling',
                region=region, magnitude=1., distributionType=UNIFORM,
                field='', localCsys=csys_cyl)
        text = ''
        text += '\n**'
        text += '\n*NODE FILE'
        text += '\nU,'
        text += '\n*MODAL FILE'
        abaqus_functions.edit_keywords(mod=mod, text=text, before_pattern=None)

    mod.Material(name='Aluminum')
    mod.materials['Aluminum'].Elastic(table=((E, nu),))
    mod.HomogeneousShellSection(name='Shell_property', preIntegrate=OFF,
            material='Aluminum', thicknessType=UNIFORM, thickness=thickness,
            thicknessField='', idealization=NO_IDEALIZATION,
            poissonDefinition=DEFAULT, thicknessModulus=None,
            temperature=GRADIENT, useDensity=OFF, integrationRule=SIMPSON,
            numIntPts=5)
    region = part.Set(faces=part.faces, name='part_faces')
    part.SectionAssignment(region=region, sectionName='Shell_property',
            offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='',
            thicknessAssignment=FROM_SECTION)
    ra.regenerate()

    mod.DisplacementBC(name='bc_top', createStepName='Initial',
            region=set_top_edges, u1=SET, u2=SET, u3=UNSET, ur1=UNSET,
            ur2=UNSET, ur3=UNSET, amplitude=UNSET, distributionType=UNIFORM,
            fieldName='', localCsys=csys_cyl)

    bottom_edges = es.getByBoundingBox(0, -2*R, -0.01*H, 2*R, 2*R, 0.01*H)
    set_bottom_edges = ra.Set(edges=bottom_edges, name='bottom_edges')
    mod.DisplacementBC(name='bc_bottom', createStepName='Initial',
            region=set_bottom_edges, u1=SET, u2=SET, u3=SET, ur1=UNSET,
            ur2=UNSET, ur3=UNSET, amplitude=UNSET, distributionType=UNIFORM,
            fieldName='', localCsys=csys_cyl)
    sidee = (es.getByBoundingBox(0, -1.01*yref, -0.01*H, R, -0.99*yref, 1.01*H)
           + es.getByBoundingBox(0, 0.99*yref, -0.01*H, R, 1.01*yref, 1.01*H))
    set_side_edges = ra.Set(edges=sidee, name='side_edges')

    u1, u2, u3, ur1, ur2, ur3 = get_bc(bc_side)
    mod.DisplacementBC(name='bc_side', createStepName='Initial',
            region=set_side_edges, u1=u1, u2=u2, u3=u3, ur1=ur1, ur2=ur2,
            ur3=ur3, amplitude=UNSET, distributionType=UNIFORM, fieldName='',
            localCsys=csys_cyl)


    if numCpus is None:
        numCpus = cpu_count()
    job = mdb.Job(name=mod_name, model=mod_name, description='', type=ANALYSIS,
            atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
            memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
            explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE,
            echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF,
            userSubroutine='', scratch='', resultsFormat=ODB,
            multiprocessingMode=DEFAULT, numCpus=numCpus, numDomains=numCpus,
            numGPUs=500)

if __name__ == '__main__':
    create_model_SPLA('test', 30, 200, 200, 1.6, 1., 10000, 71000, 0.33)
