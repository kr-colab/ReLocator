[![Documentation Status](https://readthedocs.org/projects/relocator/badge/?version=latest)](https://relocator.readthedocs.io/en/latest/?badge=latest)

# Locator

`Locator` is a supervised machine learning method for predicting the geographic origin of a sample from genotype or sequencing data. A manuscript describing it and its use can be found at https://elifesciences.org/articles/54507

## Documentation

Full documentation is available at **[https://relocator.readthedocs.io/en/latest/](https://relocator.readthedocs.io/en/latest/)**.

*   **[Installation Guide](https://relocator.readthedocs.io/en/latest/installation.html)**
*   **[CLI Usage Guide](https://relocator.readthedocs.io/en/latest/cli.html)**
*   **[Python API Guide](https://relocator.readthedocs.io/en/latest/usage.html)**

## Quick Install

### With pixi (recommended)

[Pixi](https://pixi.sh) manages all dependencies including TensorFlow and CUDA:

```bash
git clone https://github.com/kr-colab/ReLocator.git
cd ReLocator
pixi install              # GPU environment (default)
pixi install -e cpu       # CPU-only environment
pixi run test             # run tests
```

### With pip

```bash
pip install locator
```

Note: when using pip, you must manage TensorFlow and CUDA installation separately.

## License

This software is available free for all non-commercial use under the non-profit open software license v 3.0 (see LICENSE.txt).
