"""Sphinx configuration for oldiron docs."""

import os
import sys

# src layout: needed for local builds where the package is not installed.
sys.path.insert(0, os.path.abspath("../src"))

from oldiron import __version__

project = "oldiron"
author = "Paul"
release = __version__
copyright = "2026, Paul"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# MyST: generate anchors for headings so in-page links resolve.
myst_heading_anchors = 3

# Theme
html_theme = "shibuya"
html_title = "oldiron"
html_static_path = ["_static"]
html_theme_options = {
    "accent_color": "cyan",
    "github_url": "https://github.com/frolpaxa/oldiron",
    "nav_links": [
        {"title": "Command line", "url": "cli"},
        {"title": "Hardware notes", "url": "hardware"},
        {"title": "API", "url": "api"},
    ],
}
html_context = {
    "source_type": "github",
    "source_user": "frolpaxa",
    "source_repo": "oldiron",
}

# MyST (Markdown support)
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"
# rich is an optional extra; docs still build without it installed.
autodoc_mock_imports = ["rich"]
