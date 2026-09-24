#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="rule_ranker",
    version="1.0.0",
    description="GPU-accelerated Hashcat rule ranking, analysis, and CELF coverage tools",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "tqdm>=4.60",
        "pyopencl>=2021.2",
        "msgpack>=1.0",
    ],
    entry_points={
        "console_scripts": [
            "rule-ranker=run_ranker:main",
        ],
    },
)
