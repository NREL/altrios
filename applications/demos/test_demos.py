import subprocess
import os
from pathlib import Path
import pytest


demo_paths = list(Path(__file__).parent.glob("*demo*.py"))
@pytest.mark.parametrize("demo_path", demo_paths, ids=[str(dp) for dp in demo_paths])
def test_demo(demo_path: Path):
    os.environ['SHOW_PLOTS'] = "false"
    rslt = subprocess.run(
        ["python", demo_path], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )

    assert rslt.returncode == 0

