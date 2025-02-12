# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DIALECT"
copyright = "2024, Ahmed Shuaibi"
author = "Ahmed Shuaibi"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",  # Optional: Adds links to source code
    "sphinx.ext.napoleon",  # Optional: For Google-style docstrings
    "sphinx.ext.mathjax",  # Optional: For rendering math in TeX
]


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]


html_logo = "_static/logos/dark_logo_transparent_bg.png"
html_favicon = "_static/favicon.png"
html_css_files = ["custom.css"]
