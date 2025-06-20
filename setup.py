"""
Setup script for RRCE Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = []

setup(
    name="rrce-framework",
    version="0.1.0",
    author="RRCE Development Team",
    author_email="contact@rrce-framework.org",
    description="Resource-Reality Coupled Economics Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/rrce-framework",
    packages=find_packages(include=['rrce_framework', 'rrce_framework.*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
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
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "viz": [
            "dash>=2.5.0",
            "bokeh>=2.4.0",
        ],
        "parallel": [
            "dask>=2022.0.0",
            "ray>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rrce-collect-data=rrce_framework.cli:collect_data",
            "rrce-run-simulation=rrce_framework.cli:run_simulation",
            "rrce-analyze=rrce_framework.cli:analyze",
        ],
    },
    include_package_data=True,
    package_data={
        "rrce_framework": ["config/*.yaml", "data/*.json"],
    },
    zip_safe=False,
)