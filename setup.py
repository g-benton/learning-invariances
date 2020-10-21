from setuptools import setup,find_packages
import sys, os

setup(name="augerino",
      description="Learning Augmentations",
      version='0.1',
      author='Greg Benton, Marc Finzi, Pavel Izmailov',
      author_email='gwb260@nyu.edu',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['h5py',
      'olive-oil-ml @ git+https://github.com/mfinzi/olive-oil-ml',
      'torchdiffeq @ git+https://github.com/rtqichen/torchdiffeq'],#
      packages=find_packages(),#["oil",],#find_packages()
      long_description=open('README.md').read(),
)
