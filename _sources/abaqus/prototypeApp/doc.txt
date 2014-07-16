Prototype Application for Abaqus
================================

If you are willing to create ABAQUS plug-ins start with the propotype model
provided in the code, inside ``programming/abaqus/prototypeApp``, which can be
downloaded here:

`<https://github.com/saullocastro/programming/archive/master.zip>`_

It provides a straightforward way to check your new GUI. For more detail see
the ABAQUS GUI User's Manual and look for "prototypeApp".  The content of the
example package provided is::

    appIcons.py
    run_prototypeApp.bat
    prototypeApp.py
    prototypeMainWindow.py
    prototypeToolsetGui.py
    testDB.py
    testFrom.py

In the DESICOS project (see
`the documentation <http://desicos.github.io/desicos/>`_ or
`the code <https://github.com/desicos/desicos>`_) we have done a plugin for
ABAQUS based on this prototypeApp described above. It may be useful as a
reference for commands and so on...  to run this plugin execute file
"START_GUI.bat" inside the main folder

.. note::

    ABAQUS must be installed and executable with the command "abaqus cae"
