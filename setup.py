from pathlib import Path

from setuptools import find_namespace_packages, setup


def _read_requirements(file_name: str) -> list[str]:
    req_file = Path(__file__).with_name(file_name)
    if not req_file.exists():
        raise RuntimeError(f"Missing requirements file: {req_file}")

    requirements = []
    for raw in req_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)

    if not requirements:
        raise RuntimeError(f"No requirements found in {req_file}")
    return requirements


README = Path(__file__).with_name("README.md")

setup(
    name="doddlrom",
    version="1.1.0",
    packages=find_namespace_packages(include=["core*", "utils*"]),
    include_package_data=True,
    author="Dawid Kotowski",
    author_email="dkotowsk@uni-muenster.de",
    description="Deep orthogonal decomposition deep-learning ROM framework for parametric PDEs.",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    install_requires=_read_requirements("requirements.lock.txt"),
    python_requires=">=3.10,<3.15",
)
