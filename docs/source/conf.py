# -- Path setup --------------------------------------------------------------
import sys
from pathlib import Path

WS_ROOT = Path(__file__).resolve().parents[2]  # <ws>/docs/source -> <ws>

# Add BOTH the workspace src/ (for flat cases) AND each nested package directory:
sys.path.insert(0, str(WS_ROOT / "src"))  # harmless + useful if any flat pkgs exist
for pkg in ("slam_preprocessing", "slam_feature_extraction", "slam_scan_matching"):
    sys.path.insert(0, str(WS_ROOT / "src" / pkg))  # required for nested layout

# -- Project info ------------------------------------------------------------
project = "ROS 2 SLAM Demo"
author = "Your Name"

# -- Extensions --------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",     # needed for .. mermaid:: in pipeline.rst
]
autosummary_generate = True

# Mock ROS deps so docs build even without ROS installed in the venv
autodoc_mock_imports = [
    "rclpy",
    "sensor_msgs",
    "sensor_msgs_py",
    "nav_msgs",
    "geometry_msgs",
    "numpy",
]

# Pure RST (remove .md if you aren’t mixing Markdown)
source_suffix = {".rst": "restructuredtext"}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
