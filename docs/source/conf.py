# Configuration file for the Sphinx documentation builder.
# Full list of options: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup ---------------------------------------------------------------
# docs/source/conf.py
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information ------------------------------------------------------
project   = 'python OpenRecon Server'
copyright = '2026, ICM / CENIR'
author    = 'Benoit BERANGER, Solenne VINCENS'
release   = '0.1.0'

# -- General configuration ----------------------------------------------------
extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.napoleon',  
    'sphinx.ext.autosummary', 
    'sphinx.ext.viewcode', 
    'myst_parser',
    'sphinxcontrib.mermaid',
]

templates_path   = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


source_suffix = {
    '.rst': 'restructuredtext',
    '.md':  'markdown',
}

# -- Autosummary / Autodoc ----------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    'members':          True,
    'undoc-members':    True,
    'show-inheritance': True,
    'member-order':     'bysource',
}

# autodoc_mock_imports = [
#     'ismrmrd',
#     'h5py',
#     'pydicom',
#     'nibabel',
# ]

# -- Napoleon (docstrings NumPy) ----------------------------------------------
napoleon_numpy_docstring       = True
napoleon_google_docstring      = False
napoleon_include_init_with_doc = True
napoleon_use_param             = True
napoleon_use_rtype             = True

# -- HTML output --------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']

# -- MyST (Markdown) ----------------------------------------------------------
myst_heading_anchors = 3
myst_enable_extensions = ['colon_fence']
myst_fence_as_directive = ["mermaid"]
