from setuptools import setup, find_packages

setup(
    name="healthlabs_imta",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "monai",
        "pyyaml",
        "tqdm",
        "nibabel"
    ],
    author="Vincent Jaouen",
    description="HealthLabs package for TAF HEALTH, IMT Atlantique, France",
)