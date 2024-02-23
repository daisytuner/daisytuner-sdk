#!/usr/bin/env python
from setuptools import find_packages
from distutils.core import setup

with open("README.md", "r") as fp:
    long_description = fp.read()

setup(
    name="daisytuner-sdk",
    version="0.0.4",
    description="An SDK for tuning SDFGs via the Daisytuner API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Daisytuner",
    author_email="lukas.truemper@daisytuner.com",
    url="https://daisytuner.com",
    python_requires=">=3.8",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        "requests>=2.11.0",
        "tqdm>=4.64.1",
        "fire>=0.5.0",
        "numpy>=1.23.0",
        "scipy>=1.12.0",
        "dace>=0.15.0",
    ],
    extras_require={
        "dev": ["black==22.10.0", "pytest>=7.2.0", "pytest-cov>=4.1.0"],
        "profiling": ["daisytuner-likwid"],
    },
    include_package_data=True,
    package_data={
        "daisytuner": [
            "data/3rdParty/*",
        ],
    },
    classifiers=[
        "Topic :: Utilities",
    ],
    entry_points={
        "console_scripts": ["daisytuner=daisytuner.cli:main"],
    },
)
