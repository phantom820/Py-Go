from setuptools import find_packages, setup

setup(
    name='pygo',
    packages=find_packages(include=['pygo']),
    version='0.1.0',
    description='My first Python library',
    author='Tshepo Nkambule',
    license='Tshepo Nkambule',
		install_requires=['numpy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)