# setup_dp_core.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

extra = ["/O2"] if os.name == "nt" else ["-O3"]

ext = Extension(
    name="dp_core",
    sources=["dp_core.pyx"],        # if you have dp_core.c already, you can use that instead
    include_dirs=[np.get_include()],
    extra_compile_args=extra,
    # language="c++",  # <- only uncomment if the code is actually C++ and the C build fails
)

setup(
    name="dp_core",
    ext_modules=cythonize([ext], language_level="3"),
)
