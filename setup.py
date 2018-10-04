from setuptools import setup


def read(filename):
    """This is a helper function used for the reading of files into strings
    that will be used to populate certain variables in `setup`."""
    with open(filename, 'r') as f:
        return f.read()


setup(name='Basin Analysis',
      version='1.0.0-alpha',
      description='A steepest-descent gradient partitioning algorithm',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering :: Chemistry',
          'Topic :: Scientific/Engineering :: Bio-Informatics'
      ],
      url='https://github.com/jaredl7/Basin-Analysis',
      author='Jeong-Mo Choi, Jared Lalmansingh',
      author_email='jeongmochoi@wustl.edu, jared.lalmansingh@wustl.edu',
      license='MIT',
      packages=['basin_analysis'],
      install_requires=[
          'numpy',
          'matplotlib'
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      include_package_data=True,
      zip_safe=True)
