Parametric Model
================

Thanks to Dr. Mariano A. Arbelo, here it follows an example of parametric
model and a parametric study file that can be used in Abaqus.

The parametric study file can be renamed with any extension being the ``.psf``
the default extension. It is convenient to rename to ``.py`` since it uses
many of the Python commands and syntax, and this extension will allow a syntax
highlighting in many text editors:

.. literalinclude:: Example_Linear_Buckling_Parametric.py
   :language: python

Note in the Abaqus input file below how the variables defined in the
parametric study file above are used:

.. literalinclude:: Z26_L450_P4_new.inp

To submit the paramatric study::

    abaqus script=script_name

where ``script_name`` can be ``study.psf`` or ``study.py``, for example.
