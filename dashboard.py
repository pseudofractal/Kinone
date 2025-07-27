import argparse
import json
import os
import signal
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.common import add_path_arguments, load_config

base_parser = argparse.ArgumentParser(add_help=False)
base_parser = add_path_arguments(base_parser, load_config())
base_parser.add_argument("--training-pid", type=int, default=None)
cli_arguments, _ = base_parser.parse_known_args()

training_log_path = Path(cli_arguments.training_log_file)
console_log_path = Path(cli_arguments.console_log_file)
run_config_path = Path(cli_arguments.run_config_file)
stop_signal_path = Path(cli_arguments.stop_signal_file)
training_process_pid = cli_arguments.training_pid

if stop_signal_path.exists():
  stop_signal_path.unlink()

st.set_page_config(layout="wide", page_title="Kinone Training Dashboard")
st.title("Kinone Training Dashboard")

button_column_graceful, button_column_immediate = st.columns(2)
with button_column_graceful:
  if st.button("Stop after epoch", type="primary"):
    stop_signal_path.touch()
with button_column_immediate:
  if st.button("Immediate stop", type="secondary"):
    if training_process_pid:
      os.kill(training_process_pid, signal.SIGINT)
    else:
      st.error("Training PID not provided")

content_placeholder = st.empty()
run_configuration_cache = None


def read_training_rows() -> list[dict]:
  if not training_log_path.exists():
    return []
  training_rows: list[dict] = []
  with open(training_log_path, "r") as log_file_handle:
    for line in log_file_handle:
      try:
        training_rows.append(json.loads(line))
      except json.JSONDecodeError:
        continue
  return training_rows


def load_run_configuration() -> dict | None:
  if run_configuration_cache is not None:
    return run_configuration_cache
  if not run_config_path.exists():
    return None
  with open(run_config_path, "r") as cfg_handle:
    return json.load(cfg_handle)


while True:
  training_rows = read_training_rows()
  run_configuration_cache = load_run_configuration()

  with content_placeholder.container():
    logs_tab, graphs_tab = st.tabs(["Logs", "Graphs"])

    with logs_tab:
      if console_log_path.exists():
        st.code(console_log_path.read_text(), language="log", line_numbers=True)
      else:
        st.info("Console log not found")

    with graphs_tab:
      if not training_rows:
        st.info("Waiting for first epoch …")
      else:
        training_dataframe = pd.DataFrame(training_rows)
        latest_row = training_dataframe.iloc[-1]

        total_epochs = (
          run_configuration_cache["hyperparameters"].get("epochs", 0)
          if run_configuration_cache
          else 0
        )
        total_epochs = total_epochs or latest_row["epoch"]
        progress_fraction = latest_row["epoch"] / total_epochs
        st.progress(
          progress_fraction, text=f"Epoch {latest_row['epoch']} / {total_epochs}"
        )

        metric_dataframe = pd.DataFrame(
          {
            "metric": [
              "Avg Train Loss",
              "Val Loss",
              "Learning Rate",
              "Epoch Duration (s)",
            ],
            "value": [
              latest_row["average_training_loss"],
              latest_row["validation_loss"],
              latest_row["learning_rate"],
              latest_row["epoch_duration_seconds"],
            ],
          }
        ).set_index("metric")

        st.bar_chart(metric_dataframe)
        st.line_chart(
          training_dataframe,
          x="epoch",
          y=["average_training_loss", "validation_loss"],
          use_container_width=True,
        )
        st.line_chart(
          training_dataframe,
          x="epoch",
          y=["validation_auc", "validation_accuracy"],
          use_container_width=True,
        )

  time.sleep(5)
