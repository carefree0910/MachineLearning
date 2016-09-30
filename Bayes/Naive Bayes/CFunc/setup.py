from distutils.core import setup
from Cython.Build import cythonize

name = "CFunc"

setup(
    ext_modules=cythonize("{}.pyx".format(name)),
)
