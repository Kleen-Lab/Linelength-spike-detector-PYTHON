from distutils.core import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name='linelength_event_detector',
    version='0.1.0',
    author='Emma D\'Esopo',
    author_email='emma.d\'esopo@ucsf.edu',
    description='Tool for detecting spikes in EEG data using a linelength transform algorithm.',
    long_description=long_description,
    url='https://github.com/ChangLabUcsf/linelength_event_detector',
    packages=[],
    scripts=[],
    install_requires=[
        'numpy', 'scipy', 'numba', 'pytest'
    ],
)
