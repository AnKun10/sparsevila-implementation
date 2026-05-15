import sys
import subprocess
import os


def test_cli_parses_args_and_runs_help():
    result = subprocess.run(
        [sys.executable, "scripts/run_inference.py", "--help"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # project root
    )
    assert result.returncode == 0
    assert "--image" in result.stdout
    assert "--prompt" in result.stdout
    assert "--encoder-prune-ratio" in result.stdout
    assert "--decode-retrieval-ratio" in result.stdout
