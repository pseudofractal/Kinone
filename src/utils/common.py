import argparse
import json
import re
from pathlib import Path


def load_config(config_path: str | Path = "config.jsonc") -> dict:
  raw_text = Path(config_path).read_text()
  cleaned_text = re.sub(r"//.*?\n|/\*.*?\*/", "", raw_text, flags=re.S)
  return json.loads(cleaned_text)


def add_path_arguments(
  parser: argparse.ArgumentParser, config: dict | None = None
) -> argparse.ArgumentParser:
  config = config or load_config()
  path_section = config.get("paths", {})
  parser.add_argument(
    "--console-log-file",
    type=str,
    default=path_section.get("console_log_file", "console_log.txt"),
  )
  parser.add_argument(
    "--stop-signal-file",
    type=str,
    default=path_section.get("stop_signal_file", "STOP_TRAINING"),
  )
  parser.add_argument(
    "--run-config-file",
    type=str,
    default=path_section.get("run_config_file", "run_config.json"),
  )
  parser.add_argument("--training-log-file", type=str, default="training_log.json")
  return parser
