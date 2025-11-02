import subprocess, sys

def test_cli_help():
    result = subprocess.run([sys.executable, "-m", "ssfaitk.cli", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "SSF AI Toolkit CLI" in result.stdout
