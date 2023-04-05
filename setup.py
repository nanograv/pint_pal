from setuptools import setup, find_packages
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pint_pal",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Notebook/PINT-based Pulsar Timimg Analysis Software",
    author="Joe Swiggum",
    author_email="swiggumj@gmail.com",
    url="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        'ruamel.yaml',
        "pint_pulsar>=0.9.1",
        "enterprise-pulsar>=3.3.2",
        "enterprise-extensions>=v2.4.1",
        "pytest",
        "nbconvert",
        "ipywidgets>=7.6.3",
        "pypulse>=0.0.1",
        "numpy",
        "weasyprint",
        "pytest-xdist[psutil]>=2.3.0",
        "jupyter",
        "seaborn",
        "gitpython",
    ],
    python_requires=">=3.8",
)
