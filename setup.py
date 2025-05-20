from setuptools import setup, find_packages

setup(
    name="master_project_1",
    version="0.1",
    packages=find_packages(),
    author="Dawid Kotowski",
    author_email="dkotowsk@uni-muenster.de",
    description="Implements variants of the DOD-DL-ROM. Additionally some examples are provided",
    install_requires=[
        "torch>=2.0",
        "pymor>=2023.1",
        "numpy",
        "tqdm"
    ],
    python_requires=">=3.8"
)
