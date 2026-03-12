from pathlib import Path
import sys
from setuptools import setup, find_packages


def _locked_requirements():
    lock_file = Path(__file__).with_name("requirements.lock.txt")
    if not lock_file.exists():
        raise RuntimeError(f"Missing lock file: {lock_file}")

    reqs = []
    for raw in lock_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)

    if not reqs:
        raise RuntimeError(f"No requirements found in {lock_file}")
    return reqs

setup(
    name="doddlrom",
    version="0.1",
    packages=find_packages(include=["core", "utils"]),
    author="Dawid Kotowski",
    author_email="dkotowsk@uni-muenster.de",
    description="Implements variants of the DOD-DL-ROM. Additionally some examples are provided",
    install_requires=_locked_requirements(),
    python_requires="==3.9.*",
)
