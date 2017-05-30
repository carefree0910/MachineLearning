import sys
import Cython.Compiler.Main
from numpy.distutils.command import build_src
from numpy.distutils.misc_util import appendpath
from numpy.distutils import log
from os.path import join as p_join, dirname
from distutils.dep_util import newer_group
from distutils.errors import DistutilsError

try:
    import Cython.Compiler.Main
    sys.modules['Pyrex'] = Cython
    sys.modules['Pyrex.Compiler'] = Cython.Compiler
    sys.modules['Pyrex.Compiler.Main'] = Cython.Compiler.Main
    have_pyrex = True
except ImportError:
    Cython = None
    have_pyrex = False

build_src.Pyrex = Cython
build_src.have_pyrex = have_pyrex


def generate_a_pyrex_source(self, base, ext_name, source, extension):
    if self.inplace:
        target_dir = dirname(base)
    else:
        target_dir = appendpath(self.build_src, dirname(base))
    target_file = p_join(target_dir, ext_name + '.c')
    depends = [source] + extension.depends
    if self.force or newer_group(depends, target_file, 'newer'):
        import Cython.Compiler.Main
        log.info("cythonc:> %s" % target_file)
        self.mkpath(target_dir)
        options = Cython.Compiler.Main.CompilationOptions(
            defaults=Cython.Compiler.Main.default_options,
            include_path=extension.include_dirs,
            output_file=target_file)
        cython_result = Cython.Compiler.Main.compile(source, options=options)
        if cython_result.num_errors != 0:
            raise DistutilsError("%d errors while compiling %r with Cython" % (cython_result.num_errors, source))
    return target_file

build_src.build_src.generate_a_pyrex_source = generate_a_pyrex_source
