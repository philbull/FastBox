import json
import os
import sys
from setuptools import setup

# Download the Planck maps
# os.system('./scripts/get_planck_maps.sh')

setup_args = {
    'name': 'fastbox',
    'author': 'Phil Bull',
    'url': 'https://github.com/philbull/FastBox',
    'license': 'MIT',
    'version': '0.0.9',
    'description': 'Fast simulations of cosmological density fields, subject to anisotropic filtering, biasing, redshift-space distortions, foregrounds etc.',
    'packages': ['fastbox'],
    'package_dir': {'fastbox': 'fastbox'},
    'install_requires': [
        'numpy>=1.18',
        'scipy>=1.5',
        'matplotlib>=2.2',
        'sklearn',
        'scikit-image',
        'pyccl'
    ],
    'extras_require': {'fgextras': ['healpy', 'lmfit', 'multiprocessing', 'GPy']},
    'scripts': ['scripts/get_planck_maps.sh'],
    'include_package_data': True,
    'zip_safe': False
}

if __name__ == '__main__':
    setup(**setup_args)

