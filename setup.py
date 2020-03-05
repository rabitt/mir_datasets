""" Setup script for mir_datasets. """
import importlib
from setuptools import setup, find_packages

version_sfl = SourceFileLoader('mirdata.version', '../mirdata/version.py')
version_module = version_sfl.load_module()

if __name__ == '__main__':
    setup(
        name='mirdata',
        version=version_module.version,
        description='Common loaders for MIR datasets.',
        url='https://github.com/mir-dataset-loaders/mirdata',
        packages=find_packages(exclude=['test', '*.test', '*.test.*']),
        download_url='http://github.com/mir-dataset-loaders/mirdata/releases',
        package_data={'mirdata': ['indexes/*']},
        long_description="""Common loaders for MIR datasets.""",
        keywords='mir dataset loader audio',
        license='BSD-3-Clause',
        install_requires=[
            'tqdm',
            'librosa >= 0.7.0',
            'numpy>=1.16',
            'six',
            'jams',
            'requests',
        ],
        extras_require={
            'tests': [
                'pytest>=4.4.0',
                'pytest-cov>=2.6.1',
                'pytest-pep8>=1.0.0',
                'pytest-mock>=1.10.1',
                'pytest-localserver>=0.5.0',
                'testcontainers>=2.3',
                'future==0.17.1',
                'coveralls>=1.7.0',
                'dali-dataset',
            ],
            'docs': [
                'sphinx',
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
            'dali': ['DALI-dataset>=1.0.0'],
        },
    )
