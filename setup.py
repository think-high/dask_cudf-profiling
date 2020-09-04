import os

__location__ = os.path.dirname(__file__)

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# read the contents of README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dask-cudf-profiling',
    version='1.0.0',
    author='Rahul Bhojwani',
    author_email='rahulbhojwani2003@gmail.com',
    packages=['dask_cudf_profiling'],
    url='https://github.com/think-high/dask_cudf-profiling',
    license='MIT',
    description='Generate profile report for dask cudf DataFrame',
    install_requires=[
        # "pandas>=0.19",
        "dask>=2.11.0",
        "matplotlib>=1.4",
        "jinja2>=2.8",
        "six>=1.9",
        "scipy>=1.4.1"
    ],
    include_package_data = True,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Framework :: IPython',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'

    ],
    keywords='dask cudf data-science data-analysis python jupyter ipython',
    long_description=long_description,
    long_description_content_type='text/markdown'

)
