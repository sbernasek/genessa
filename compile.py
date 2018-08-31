from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

import sys, os
modules_path = '../'
if modules_path not in sys.path:
    sys.path.insert(0, modules_path)

ext_modules = [
    Extension("solver.rxns", ["solver/rxns.pyx"], include_dirs=['.']),
    Extension("solver.cyRNG", ["solver/cyRNG.pyx", "solver/SimpleRNG.cpp"], language="c++", include_dirs=['.']),
    Extension("solver.ssa", ["solver/ssa.pyx"], include_dirs=['.']),
    Extension("solver.signals", ["solver/signals.pyx"], include_dirs=['.']),
    ]

setup(
  name='solver_tools',
  cmdclass={'build_ext': build_ext},
  ext_modules=ext_modules,
  script_args=['build_ext'],
  options={'build_ext':{'inplace':True, 'force':True}},
  include_dirs=[get_include()]
)

print('******** CYTHON COMPILATION COMPLETE ******')

