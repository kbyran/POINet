import os
import re
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    content = None
    with open(os.path.join(here, *parts), 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="poi",
    version=find_version("poi", "__init__.py"),
    description="A flexible computer vision library on human analysis.",
    author="kbyran",
    author_email="im@kbyran.com",
    license='Apache-2.0',
    packages=find_packages(),
)
