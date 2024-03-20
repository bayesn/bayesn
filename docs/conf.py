import os
import sys
sys.path.insert(0, os.path.abspath('../'))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'bayesn'
copyright = '2023, Matthew Grayling, Stephen Thorp, Kaisey Mandel'
author = 'Matthew Grayling, Stephen Thorp, Kaisey Mandel'
release = '0.3.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['numpydoc', 'sphinx.ext.autosummary', 'sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
autosummary_generate = True
autosummary_imported_members = True
