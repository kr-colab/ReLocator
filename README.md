[![Documentation Status](https://readthedocs.org/projects/relocator/badge/?version=latest)](https://relocator.readthedocs.io/en/latest/?badge=latest)

# Locator

`Locator` is a supervised machine learning method for predicting the geographic origin of a sample from genotype or sequencing data. A manuscript describing it and its use can be found at https://elifesciences.org/articles/54507

## Documentation

Full documentation is available at **[https://relocator.readthedocs.io/en/latest/](https://relocator.readthedocs.io/en/latest/)**.

*   **[Installation Guide](https://relocator.readthedocs.io/en/latest/installation.html)**
*   **[CLI Usage Guide](https://relocator.readthedocs.io/en/latest/cli.html)**
*   **[Python API Guide](https://relocator.readthedocs.io/en/latest/usage.html)**

## Quick Install

The easiest way to install `relocator` is to download the github repo and run the setup script. It's usually a good idea to do this in a new conda environment:

```bash
conda create --name locator
conda activate locator
git clone https://github.com/kr-colab/relocator.git
cd locator
pip install .
```

## License

This software is available free for all non-commercial use under the non-profit open software license v 3.0 (see LICENSE.txt).
