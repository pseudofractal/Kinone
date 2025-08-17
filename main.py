import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import train
from src.data.nih_datamodule import NIHDataModule
from src.utils.common import add_path_arguments, load_config

config = load_config()

path_config = config.get("paths", {})
train_config = config.get("train", {})
dashboard_config = config.get("dashboard", {})

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=train_config.get("dataset", "nih"))
parser.add_argument("--epochs", type=int, default=train_config.get("epochs", 10))
parser.add_argument(
  "--batch-size", type=int, default=train_config.get("batch_size", 32)
)
parser.add_argument("--lr", type=float, default=train_config.get("lr", 1e-3))
parser.add_argument(
  "--lr-step-size", type=int, default=train_config.get("lr_step_size", 5)
)
parser.add_argument("--lr-gamma", type=float, default=train_config.get("lr_gamma", 0.1))
parser.add_argument(
  "--maximum-gradient-norm", type=float, default=train_config.get("maximum_gradient_norm", None)
)
parser.add_argument("--seed", type=int, default=train_config.get("seed", 42))
parser.add_argument(
  "--load-checkpoint", type=str, default=train_config.get("load_checkpoint", None)
)
parser.add_argument("--export-onnx", type=str, default=None)
parser = add_path_arguments(parser, config)

dashboard_group = parser.add_mutually_exclusive_group()
dashboard_group.add_argument("--dashboard", dest="dashboard", action="store_true")
dashboard_group.add_argument("--no-dashboard", dest="dashboard", action="store_false")
parser.set_defaults(dashboard=dashboard_config.get("enabled", True))

parser = NIHDataModule.add_argparse_args(parser)
args = parser.parse_args()

run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_directory = Path("runs") / run_timestamp
run_directory.mkdir(parents=True, exist_ok=True)
shutil.copyfile("config.jsonc", run_directory / "config.jsonc")
args.run_directory = run_directory
args.console_log_file = run_directory / Path(args.console_log_file).name
args.stop_signal_file = run_directory / Path(args.stop_signal_file).name
args.run_config_file = run_directory / Path(args.run_config_file).name
args.training_log_file = run_directory / Path(args.training_log_file).name

dashboard_proc = None
try:
  if args.dashboard:
    dashboard_cmd = [
      sys.executable,
      "-m",
      "streamlit",
      "run",
      "dashboard.py",
      "--",
      "--training-pid",
      str(os.getpid()),
      "--console-log-file",
      str(args.console_log_file.resolve()),
      "--stop-signal-file",
      str(args.stop_signal_file.resolve()),
      "--run-config-file",
      str(args.run_config_file.resolve()),
      "--training-log-file",
      str(args.training_log_file.resolve()),
    ]
    dashboard_proc = subprocess.Popen(dashboard_cmd)

  train.train(args)
finally:
  if dashboard_proc is not None:
    dashboard_proc.terminate()
    try:
      dashboard_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
      dashboard_proc.kill()
