from setuptools import setup, find_packages

setup(
    name='src',
    version='1.0.0',
    author='Anonymized',
    url='',
    packages=['src/']+find_packages(),
    zip_safe=False,
    include_package_data=True
)