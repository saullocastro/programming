from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [Extension('my_cuda',
                   sources=['my_cuda.pyx'],
                   libraries=['cuda_blockadd', 'cuda_threadadd',
                              'cuda_btadd', 'cuda_longadd'],
                   language='c',
                   extra_compile_args=['/openmp',
                       '/O2', '/favor:INTEL64', '/fp:fast'],
                   extra_link_args=[],
                   )]
setup(name = 'my_cuda',
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules)
