from distutils.core import setup, Extension
import numpy as np

#define the extension module
softsubspace_module = Extension(
    'softsubspace',
    sources=["softsubspacemodule.c", "kmeans.c", "utils.c"],
    include_dirs=[np.get_include()]
)

#run the setup
setup(name="softsubspace",
      description="Soft Subspace Module",
      ext_modules=[softsubspace_module])

