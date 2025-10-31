"""
IntakeSIM: Air-Breathing Electric Propulsion Particle Simulation

Particle-based simulation (DSMC + PIC) for ABEP system validation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="intakesim",
    version="0.1.0",
    author="AeriSat Systems",
    author_email="info@aerisat.com",
    description="Particle simulation for Air-Breathing Electric Propulsion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AeriSat/IntakeSIM",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "numba>=0.58.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)
