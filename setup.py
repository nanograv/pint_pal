from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="timing_analysis",
    version="0.1.0",
    description="NANOGrav Timimg Analysis",
    author="Joe Swiggum",
    author_email="swiggumj@gmail.com",
    url="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="./src/*"),
    python_requires='>=3.6'
)
