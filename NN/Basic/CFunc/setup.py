# encoding: utf8

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
  Extension('core', ['core.pyx'],
            include_dirs=[numpy.get_include()]
  ),
]

setup(
    ext_modules=cythonize(extensions),
)
