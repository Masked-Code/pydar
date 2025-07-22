"""Setup configuration for PyDar radar simulator."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pydar",
    version="0.1.0",
    author="PyDar Development Team",
    author_email="pydar@example.com",
    description="A high-fidelity radar simulator written in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pydar",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=0.990",
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "examples": [
            "jupyter>=1.0",
            "notebook>=6.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "pydar=pydar.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pydar": ["data/*.json", "data/*.csv"],
    },
    zip_safe=False,
)
