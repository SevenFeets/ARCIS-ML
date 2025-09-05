#!/usr/bin/env python3
"""
ARCIS Package Setup

Setup script for the ARCIS weapon detection system.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="arcis",
    version="1.0.0",
    author="ARCIS Team",
    author_email="team@arcis.ai",
    description="Advanced Real-time Computer Intelligence System for Weapon Detection",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/arcis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.2.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        "edge": [
            "tensorflow-lite>=2.13.0",
            "onnxruntime>=1.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "arcis-train=arcis.scripts.train:main",
            "arcis-inference=arcis.scripts.inference:main",
            "arcis-evaluate=arcis.scripts.evaluate:main",
            "arcis-deploy=arcis.scripts.deploy:main",
        ],
    },
    include_package_data=True,
    package_data={
        "arcis": [
            "configs/*.yaml",
            "configs/*.yml",
            "models/*.pt",
            "models/*.onnx",
            "models/*.tflite",
        ],
    },
    zip_safe=False,
)
