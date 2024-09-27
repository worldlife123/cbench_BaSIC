from setuptools import setup, find_packages
from glob import glob
import os
import sys
import sysconfig

# TODO: pybind c++ code compile
import setuptools
from setuptools import Extension, Distribution
from sys import platform
from subprocess import call
from multiprocessing import cpu_count

# NOTE: library not found libc10 on macOS when using torch extensions!
# from torch.utils.cpp_extension import BuildExtension, CppExtension
# from torch.utils.cpp_extension import CudaExtension
from pybind11.setup_helpers import Pybind11Extension, build_ext
# from setup_helpers import Pybind11Extension, build_ext

ZSTD_CBENCH_PATH = "3rdparty/zstd-cbench"

class build_ext_zstd(build_ext):
    # adjusted from https://github.com/Turbo87/py-xcsoar/blob/master/setup.py
    def run(self):
        # build zstd
        build_path = os.path.abspath(self.build_temp)

        cmd = [
            'make',
            # 'OUT=' + build_path,
            'V=' + str(self.verbose),
        ]

        try:
            cmd.append('-j%d' % cpu_count())
        except NotImplementedError:
            print('Unable to determine number of CPUs. Using single threaded make.')

        options = [
            # 'DEBUGLEVEL=6',
            # # 'BACKTRACE=1',
            # 'ZSTD_LIB_MINIFY=1',
            # 'FSE_API=0',
            # 'ZSTD_CUSTOM_DICT=0',
        ]
        cmd.extend(options)

        targets = ['lib']
        cmd.extend(targets)

        # fix for macos (apple m1/arm platform)
        if platform == 'darwin':
            cmd.append('CC=gcc --target=x86_64-apple-macos -g')
        # linux requires PIC to link static libs
        elif platform == 'linux':
            cmd.append('CC=gcc -fPIC -g')

        def compile():
            call(cmd, cwd=ZSTD_CBENCH_PATH)

        self.execute(compile, [], 'Compiling zstd')

        # run original build code
        build_ext.run(self)

zstd_wrapper_sources = [
    # "3rdparty/FiniteStateEntropy/lib",
    "cbench/csrc/zstd_wrapper.cpp",
]

ext_modules = dict(
    # zstd_wrapper=Pybind11Extension(
    #     "cbench.zstd_wrapper",
    #     zstd_wrapper_sources,
    #     include_dirs=['.'],
    #     # library_dirs=library_dirs,
    #     # libraries=libraries,
    #     extra_compile_args=['-g'], # enable debugging symbols
    #     # extra_link_args=['-static'],
    #     extra_objects=[os.path.join(ZSTD_CBENCH_PATH, "lib/libzstd.a")], # for static link
    # ),
    # tans=Pybind11Extension(
    #     name=f"cbench.tans",
    #     sources=["cbench/csrc/tans/tans_interface.cpp",],
    #     language="c++",
    #     include_dirs=["cbench/csrc/tans", "."],
    #     extra_compile_args=['-std=c++17', '-g'], # enable debugging symbols
    #     extra_objects=[os.path.join(ZSTD_CBENCH_PATH, "lib/libzstd.a")], # for static link
    # ),
    rans=Pybind11Extension(
        name=f"cbench.rans",
        sources=["cbench/csrc/rans/rans_interface.cpp",],
        language="c++",
        include_dirs=["cbench/csrc/rans"],
        extra_compile_args=['-std=c++17', '-g'], # enable debugging symbols
        # extra_link_args=['-L/usr/lib/x86_64-linux-gnu/'] # this may solve link error
    ),
    ar=Pybind11Extension(
        name=f"cbench.ar",
        sources=glob("cbench/csrc/ar/*.cpp"),
        language="c++",
        include_dirs=["cbench/csrc/ar"],
        extra_compile_args=['-std=c++17', '-g'], # enable debugging symbols
        # extra_link_args=['-L/usr/lib/x86_64-linux-gnu/'] # this may solve link error
    ),
    ans=Pybind11Extension(
        name=f"cbench.ans",
        sources=glob("cbench/csrc/ans/*.cpp"),
        language="c++",
        include_dirs=["cbench/csrc/ans"],
        extra_compile_args=['-std=c++17', '-g'], #, '-O0'], # enable debugging symbols
        # extra_link_args=['-L/usr/lib/x86_64-linux-gnu/'] # this may solve link error
    ),
)

cmdclass = {
    'build_ext': build_ext, # build_ext_zstd,
}

# skip check requires to speed up installation, but may be needed for production
install_requires = [
    # # basic
    # "numpy",
    # "scipy",
    # # pytorch
    # "torch",
    # "torchvision",
    # "pytorch-lightning",
    # # compression libs
    # "zstandard",
    # "brotlipy",
    # # other
    # # "cython", # cython needed to be installed beforehand for pandas
    # "tqdm",
    # "pandas",
]

def generate_pyi(module_name):
    import torch
    from pybind11_stubgen import ModuleStubsGenerator

    module = ModuleStubsGenerator("cbench."+module_name)
    module.parse()
    module.write_setup_py = False

    with open("cbench/%s.pyi" % module_name, "w") as fp:
        fp.write("#\n# Automatically generated file, do not edit!\n#\n\n")
        fp.write("\n".join(module.to_lines()))

# TODO: 'CC=clang CXX=clang++' on macos ?
setup(
    name="cbench",
    version="0.2",
    description="Data Compression Benchmark",
    url="",
    packages=find_packages(),
    install_requires=install_requires,
    ext_modules=list(ext_modules.values()),
    cmdclass=cmdclass,
)


# test if the extension is built correctly
import importlib
import traceback
for ext in ext_modules.values():
    print(f"Try import {ext.name}")
    try:
        importlib.import_module(ext.name)
    except ImportError as e:
        print(f"Import {ext.name} failed!")
        traceback.print_exc()
        exit(1)
    print(f"Import {ext.name} success!")
print("Setup success!")

for module_name in ext_modules.keys():
    generate_pyi(module_name)