import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RL-PETS",
    version="0.0.1",
    author="Erfan Azad",
    author_email="basiri@cs.uni-freiburg.de",
    description="Minimal PETS in Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Erfi/PETS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)