import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    REQUIRED_PKGS = f.read().splitlines()

# TODO(LSR): set author, email, url once paper has been published.
setuptools.setup(
    name="rpca-denise",
    version="0.0.1",
    author="C. Herrera, F. Krach, A. Kratsios, P. Ruyssen, J. Teichmann",,
    author_email="calypso.herrera@math.ethz.ch, florian.krach@me.com",
    description="Code used to runs experiments for Denise paper.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepRPCA/Denise",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
