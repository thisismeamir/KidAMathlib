from setuptools import find_packages, setup

setup(
    name='KidAMAthlib',
    packages= find_packages(),
    version='1.1.0',
    description='Kid A Private mathematical methods need for his researches.',
    author='KID A',
    license='None',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)