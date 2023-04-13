import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RL_practice",
    version="1.0.0",
    author="Donghyun Min",
    author_email="mdh38112@sogang.ac.kr",
    description="A package for reinforcement learning practice",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", include=["*"]),
    python_requires=">= 3.8",
    install_requires=["numpy == 1.22.3", "importlib-resources == 5.9.0"],
)
