"""
Configuration settings for Social Network Analysis (22ITPA0) Project.
Centralizes random seeds, dataset selection, and directory paths.
"""

import os
from pathlib import Path

# Resolve project root dynamically (works from src/, notebooks/, or root)
def get_project_root() -> Path:
    current = Path(__file__).resolve().parent
    if current.name == "src" or current.name == "notebooks":
        return current.parent
    # If running from project root
    if (current / "src").exists() and (current / "data").exists():
        return current
    # Fallback to parent
    return current.parent

PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Global random seed for reproducible results across all analyses
RANDOM_SEED = 42

# Active Dataset Selection:
# Options: 'peertube' (Fedivertex PeerTube Follow network) or 'wiki_vote' (SNAP wiki-Vote)
# Default is 'peertube' as provided in the repository workspace.
ACTIVE_DATASET = os.environ.get("SNA_DATASET", "peertube").lower()

DATASET_CONFIGS = {
    "peertube": {
        "name": "Fedivertex PeerTube Follow Network",
        "raw_path": RAW_DATA_DIR / "peertube" / "interactions.csv",
        "metadata_path": RAW_DATA_DIR / "peertube" / "instances.csv",
        "source_col": "Source",
        "target_col": "Target",
        "has_header": True,
        "delimiter": ",",
        "node_type": "PeerTube Instance Server",
        "edge_type": "Directed Follow Link",
        "citation": "Damie & Cyffers, WWW 2026 (arXiv:2505.20882)",
    },
    "wiki_vote": {
        "name": "SNAP Wikipedia Who-votes-on-whom Network",
        "raw_path": RAW_DATA_DIR / "wiki_vote" / "wiki-Vote.txt",
        "metadata_path": None,
        "source_col": "FromNodeId",
        "target_col": "ToNodeId",
        "has_header": False,
        "delimiter": "\t",
        "node_type": "Wikipedia User",
        "edge_type": "Directed RfA Vote",
        "citation": "Leskovec, Huttenlocher, Kleinberg, WWW 2010",
    }
}
