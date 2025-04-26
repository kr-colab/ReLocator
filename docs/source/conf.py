import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "Locator"
copyright = "2025, Kern-Ralph  Co-Lab"
author = "Kern-Ralph  Co-Lab"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Add intersphinx mapping for external libraries
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    #'tensorflow': ('https://www.tensorflow.org/api_docs/python', None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# Templates path
templates_path = ["_templates"]
exclude_patterns = []

# HTML theme settings
html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]

# Add any Sphinx extension module names here
autosummary_generate = True  # Generate stub pages for autosummary
add_module_names = False  # Remove module names from generated documentation

# Make sure the target directory exists
import os

if not os.path.exists("_autosummary"):
    os.makedirs("_autosummary")
