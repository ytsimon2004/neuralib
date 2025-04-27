# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NeuraLib'
copyright = '2024, Yu-Ting Wei'
author = 'Yu-Ting Wei'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.viewcode',
              'sphinx_tabs.tabs',
              'sphinx_prompt',
              'sphinx_copybutton',
              'nbsphinx']

templates_path = ['_templates']
exclude_patterns = ['**/site-packages/**']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {

}

# -- Options for autodoc ------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_class_signature = 'separated'
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'inherited-members': True,
    'show-inheritance': True,
}

# -- Options for autosummary ------------------------------------------------
autosummary_generate = True


suppress_warnings = ["autodoc.duplicate"]

# -- Options for nbsphinx -------------------------------------
nbsphinx_execute = 'never'
