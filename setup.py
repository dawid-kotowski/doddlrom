from setuptools import setup, find_packages

setup(
    name="master_project_1",
    version="0.1",
    packages=find_packages(),  # This will automatically find all packages (directories with an __init__.py)
    author="Dawid Kotowski",
    author_email="dkotowsk@uni-muenster.de",
    description="Implements variants of the DOD-DL-ROM. Additionally some examples are provided",
)
