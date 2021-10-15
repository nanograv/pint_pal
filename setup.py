from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="timing_analysis",
    version="1.0.0",
    description="NANOGrav Timimg Analysis",
    author="Joe Swiggum",
    author_email="swiggumj@gmail.com",
    url="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir = {'': 'src'},
    packages = find_packages('src'),
    install_requires=[
        'ruamel.yaml',
        "pint_pulsar>=0.8.3",
        "enterprise-pulsar>=3.1.0",
        "enterprise-extensions==v2.1.0",
        "pytest",
        "nbconvert",
        "ipywidgets>=7.6.3",
        "pypulse>=0.0.1"
        "numpy"
        "pytest-xdist[psutil]>=2.3.0"
],
    python_requires='>=3.6'
)
