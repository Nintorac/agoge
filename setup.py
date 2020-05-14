#!/usr/bin/env python
from setuptools import setup, find_packages
from pathlib import Path

readme = Path('README.md').read_text()
requirements = Path('requirements.txt').read_text().split('\n')
__version__ = Path('version').read_text().strip()

setup(
    name='agoge',
    packages = ['agoge'],
    license='MIT',
    version=__version__,
    author='Nintorac',
    author_email='agoge@nintorac.dev',
    url='https://github.com/nintorac/agoge',
    description='Machine Learning infra',
    download_url = 'https://github.com/Nintorac/agoge/archive/0.0.1.tar.gz',
    long_description=readme,
    zip_safe=True,
      keywords = ['pytorch', 'machine learning'],
    install_requires=requirements,
    classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
  long_description_content_type='text/markdown'
)