from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="locator",
    version="1.2.1",
    description="supervised machine learning of geographic location from genetic variation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kr-colab/locator",
    license="NPOSL-3.0",
    packages=find_packages(exclude=[]),
    install_requires=[
        "numpy>=1.26.0",
        "tensorflow; platform_system=='Darwin'",
        "tf-nightly[and-cuda]; platform_system=='Linux' or platform_system=='Windows'",
        "h5py",
        "scikit-allel",
        "scikit-learn",
        "matplotlib",
        "scipy",
        "tqdm",
        "pandas",
        "zarr<3.0.0",
        "seaborn",
        # geospatial
        "cartopy",
        "geopy",
        "geopandas",
        "rasterio",
        "rasterstats",
        "shapely",
        "affine",
        # Optional dependencies
        "jupyter",
        "ipykernel",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",  # Code formatting
            "flake8",  # Code linting
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx_rtd_theme",
            "sphinx-autodoc-typehints",
            "sphinx-autobuild",
        ],
    },
    entry_points={
        "console_scripts": [
            "locator=locator.cli:main",
            "vcf_to_zarr=scripts.vcf_to_zarr:main",
        ],
    },
    zip_safe=False,
    python_requires=">=3.8,<3.13",
    setup_requires=["numpy>=1.20.0"],
    author="Andrew Kern",
    author_email="adkern@uoregon.edu",
    project_urls={
        "Bug Reports": "https://github.com/kr-colab/locator/issues",
        "Documentation": "https://github.com/kr-colab/locator#readme",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Non-Profit Open Software License 3.0 (NPOSL-3.0)",
    ],
)
