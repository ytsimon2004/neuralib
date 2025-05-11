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
              'sphinx_copybutton',
              'nbsphinx']

templates_path = ['_templates']
exclude_patterns = ['**/site-packages/**']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_context = {
    "display_github": True,
    "github_user": "ytsimon2004",
    "github_repo": "neuralib",
    "github_version": "main",
    "conf_py_path": "/doc/source/",  # Path in the checkout to the docs root
}

# -- Options for autodoc ------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_class_signature = 'separated'
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'inherited-members': False,
    'show-inheritance': True,
}

# -- Options for autosummary ------------------------------------------------
autosummary_generate = True

# -- Options for nbsphinx -------------------------------------
nbsphinx_execute = 'never'

# -- Options for Copy-button settings --------------------------
copybutton_prompt_text = r'^\$ '
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True
