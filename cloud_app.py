"""Streamlit Community Cloud entry point. Local launcher still uses app.py."""
import os
import runpy
from pathlib import Path

os.environ["DETOX_CLOUD"] = "1"
os.environ["DETOX_LOW_MEMORY"] = "1"
runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
