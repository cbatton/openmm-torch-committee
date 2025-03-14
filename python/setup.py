from setuptools import setup, Extension
import os
import platform

version = '@OPENMM_TORCH_COMMITTEE_VERSION@'
openmm_dir = '@OPENMM_DIR@'
torch_include_dirs = '@TORCH_INCLUDE_DIRS@'.split(';')
nn_plugin_header_dir = '@NN_PLUGIN_HEADER_DIR@'
nn_plugin_library_dir = '@NN_PLUGIN_LIBRARY_DIR@'
torch_dir, _ = os.path.split('@TORCH_LIBRARY@')

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++17']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.13']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.13']

extension = Extension(name='_openmmtorchcommittee',
                      sources=['TorchPluginWrapper.cpp'],
                      libraries=['OpenMM', 'OpenMMTorchCommittee'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), nn_plugin_header_dir] + torch_include_dirs,
                      library_dirs=[os.path.join(openmm_dir, 'lib'), nn_plugin_library_dir],
                      runtime_library_dirs=[os.path.join(openmm_dir, 'lib'), torch_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='openmmtorchcommittee',
      version=version,
      py_modules=['openmmtorchcommittee'],
      ext_modules=[extension],
      install_requires=['openmm', 'torch']
     )
