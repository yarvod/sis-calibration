from setuptools import setup, find_packages
from io import open


def read(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


setup(
    name="sis-calibration",
    version="0.24",
    description="SIS calibration",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="sisteralab",
    author_email="ya.vodzyanovskiy@lebedev.ru",
    url="https://github.com/sisteralab/sis-calibration",
    keywords="sis vna",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "mpmath", "matplotlib", "port_calibration"],
)
