"""Keep small CPU checks fast and temporary test files inside the workspace."""
import tempfile
from pathlib import Path

import torch


torch.set_num_threads(2)
workspace_temp = Path(__file__).resolve().parents[1]/".test-tmp"
workspace_temp.mkdir(exist_ok=True)
tempfile.tempdir = str(workspace_temp)
