# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys


def get_version():
    namespace = {}
    exec(open("../../../torchdatasets/_version.py").read(), namespace)
    return namespace["__version__"]


sys.path.insert(0, os.path.abspath("../../.."))


# -- Project information -----------------------------------------------------

project = "torchdatasets"
copyright = "2019, Szymon Maszke"
author = "Szymon Maszke"
version = get_version()
release = version


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
]

# Heavy runtime deps are mocked so autodoc can import the package without
# installing torch / torchvision in the docs environment.
autodoc_mock_imports = ["torch", "torchvision", "numpy"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
}

default_role = "py:obj"  # reference Python objects by default
