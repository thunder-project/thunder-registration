#!/usr/bin/env python

from setuptools import setup

version = '1.0.1'

required = open('requirements.txt').read().split('\n')

setup(
    name='thunder-registration',
    version=version,
    description='algorithms for image registration',
    author='freeman-lab',
    author_email='the.freeman.lab@gmail.com',
    url='https://github.com/thunder-project/thunder-registration',
    packages=[
        'registration',
        'registration.algorithms'
    ],
    install_requires=required,
    long_description='See ' + 'https://github.com/thunder-project/thunder-registration',
    license='MIT'
)
