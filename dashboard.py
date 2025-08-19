import argparse
import json
import os
import signal
import time
from pathlib import Path

import pandas
import streamlit as st

from src.utils.common import add_path_arguments, load_config


def format_seconds_to_human_readable_string(total_seconds: float) -> str:
  minutes, seconds = divmod(total_seconds, 60)
  hours, minutes = divmod(minutes, 60)
  return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def read_training_log_rows(training_log_file_path: Path) -> list[dict]:
  if not training_log_file_path.exists():
    return []
  try:
    with open(training_log_file_path, "r") as log_file_handle:
      file_content = log_file_handle.read()
      if not file_content:
        return []
      return json.loads(file_content)
  except (json.JSONDecodeError, FileNotFoundError):
    return []


def load_run_configuration(run_configuration_file_path: Path) -> dict | None:
  if not run_configuration_file_path.exists():
    return None
  with open(run_configuration_file_path, "r") as configuration_file_handle:
    return json.load(configuration_file_handle)


def display_run_data(training_log_rows: list[dict], run_configuration: dict | None):
  if not training_log_rows:
    st.info("Waiting for the first epoch to complete...")
    return

  training_dataframe = pandas.DataFrame(training_log_rows)
  latest_epoch_data = training_dataframe.iloc[-1]
  best_area_under_curve_epoch_data = training_dataframe.loc[
    training_dataframe["validation_auc"].idxmax()
  ]

  total_number_of_epochs = (
    run_configuration.get("hyperparameters", {}).get("epochs", 0)
    if run_configuration
    else latest_epoch_data["epoch"]
  )

  if latest_epoch_data["epoch"] >= total_number_of_epochs:
    st.success(f"Training finished after {total_number_of_epochs} epochs.")

  st.header("Live Metrics")
  st.progress(
    latest_epoch_data["epoch"] / total_number_of_epochs,
    text=f"Epoch {latest_epoch_data['epoch']} / {total_number_of_epochs}",
  )

  key_performance_indicator_columns = st.columns(6)
  area_under_curve_delta = (
    latest_epoch_data["validation_auc"]
    - best_area_under_curve_epoch_data["validation_auc"]
  )
  key_performance_indicator_columns[0].metric(
    label="Validation AUC",
    value=f"{latest_epoch_data['validation_auc']:.4f}",
    delta=f"{area_under_curve_delta:.4f} vs Best",
  )

  validation_loss_delta = None
  if len(training_dataframe) > 1:
    previous_validation_loss = training_dataframe.iloc[-2]["validation_loss"]
    validation_loss_delta = (
      latest_epoch_data["validation_loss"] - previous_validation_loss
    )
  key_performance_indicator_columns[1].metric(
    label="Validation Loss",
    value=f"{latest_epoch_data['validation_loss']:.4f}",
    delta=f"{validation_loss_delta:.4f}" if validation_loss_delta is not None else None,
    delta_color="inverse",
  )

  key_performance_indicator_columns[2].metric(
    label="Best Validation AUC",
    value=f"{best_area_under_curve_epoch_data['validation_auc']:.4f}",
    help=f"Achieved at epoch {best_area_under_curve_epoch_data['epoch']}.",
  )

  key_performance_indicator_columns[3].metric(
    label="Last Epoch Duration",
    value=format_seconds_to_human_readable_string(
      latest_epoch_data["epoch_duration_seconds"]
    ),
  )

  average_epoch_duration_seconds = training_dataframe["epoch_duration_seconds"].mean()
  key_performance_indicator_columns[4].metric(
    label="Average Epoch Duration",
    value=format_seconds_to_human_readable_string(average_epoch_duration_seconds),
  )

  remaining_epochs = total_number_of_epochs - latest_epoch_data["epoch"]
  estimated_time_remaining_seconds = remaining_epochs * average_epoch_duration_seconds
  key_performance_indicator_columns[5].metric(
    label="Est. Time Remaining",
    value=format_seconds_to_human_readable_string(estimated_time_remaining_seconds),
  )

  st.subheader("Loss Curves")
  st.line_chart(
    training_dataframe.rename(
      columns={
        "average_training_loss": "Training Loss",
        "validation_loss": "Validation Loss",
      }
    ),
    x="epoch",
    y=["Training Loss", "Validation Loss"],
  )

  st.subheader("Performance Metrics")
  st.line_chart(
    training_dataframe.rename(
      columns={
        "validation_auc": "Validation AUC",
        "validation_accuracy": "Validation Accuracy",
      }
    ),
    x="epoch",
    y=["Validation AUC", "Validation Accuracy"],
  )


st.set_page_config(layout="wide", page_title="Kinone Dashboard")

argument_parser = argparse.ArgumentParser(add_help=False)
argument_parser.add_argument("--training-pid", type=int, default=None)
command_line_arguments, _ = argument_parser.parse_known_args()
TRAINING_PROCESS_IDENTIFIER = command_line_arguments.training_pid

if TRAINING_PROCESS_IDENTIFIER:
  st.title("Kinone Live Training Dashboard")

  path_argument_parser = add_path_arguments(
    argparse.ArgumentParser(add_help=False), load_config()
  )
  path_arguments, _ = path_argument_parser.parse_known_args()
  TRAINING_LOG_FILE_PATH = Path(path_arguments.training_log_file)
  CONSOLE_LOG_FILE_PATH = Path(path_arguments.console_log_file)
  RUN_CONFIGURATION_FILE_PATH = Path(path_arguments.run_config_file)
  STOP_SIGNAL_FILE_PATH = Path(path_arguments.stop_signal_file)

  if "stop_signal_sent" not in st.session_state:
    st.session_state.stop_signal_sent = False

  graceful_stop_button_column, immediate_stop_button_column = st.columns(2)
  if graceful_stop_button_column.button("Stop after epoch", type="primary"):
    STOP_SIGNAL_FILE_PATH.touch()
    st.session_state.stop_signal_sent = True
    st.toast("Graceful stop signal sent.")
  if immediate_stop_button_column.button("Immediate stop", type="secondary"):
    os.kill(TRAINING_PROCESS_IDENTIFIER, signal.SIGINT)
    st.toast("Immediate stop signal sent.")

  run_configuration = load_run_configuration(RUN_CONFIGURATION_FILE_PATH)
  with st.sidebar:
    st.header("Run Configuration")
    if run_configuration:
      st.json(run_configuration["hyperparameters"], expanded=False)
      st.json(run_configuration["dataset_information"])
    else:
      st.info("Waiting for run_config.json...")

  main_content_placeholder = st.empty()
  while True:
    training_log_rows = read_training_log_rows(TRAINING_LOG_FILE_PATH)
    with main_content_placeholder.container():
      if st.session_state.stop_signal_sent:
        st.warning("Stop signal sent. Training will halt after the current epoch.")

      graphs_tab, logs_tab = st.tabs(["📈 Graphs & Metrics", "📄 Logs"])
      with graphs_tab:
        display_run_data(training_log_rows, run_configuration)
      with logs_tab:
        st.subheader("Console Logs")
        if CONSOLE_LOG_FILE_PATH.exists():
          with open(CONSOLE_LOG_FILE_PATH, "r") as file_handle:
            log_lines = file_handle.readlines()
            st.code("".join(log_lines[-50:]), language="log")
            with st.expander("Show full console log"):
              st.code("".join(log_lines), language="log", line_numbers=True)
        else:
          st.info("Console log not found.")
    time.sleep(1)

else:
  st.title("Kinone Training Analysis")

  run_directories = sorted(
    [directory for directory in Path("runs").iterdir() if directory.is_dir()],
    key=os.path.getmtime,
    reverse=True,
  )

  if not run_directories:
    st.error("No training runs found in the 'runs' directory.")
  else:
    selected_run_name = st.selectbox(
      "Select a training run to inspect:", options=[d.name for d in run_directories]
    )
    selected_run_directory = Path("runs") / selected_run_name

    TRAINING_LOG_FILE_PATH = selected_run_directory / "training_log.json"
    CONSOLE_LOG_FILE_PATH = selected_run_directory / "console_log.txt"
    RUN_CONFIGURATION_FILE_PATH = selected_run_directory / "run_config.json"

    run_configuration = load_run_configuration(RUN_CONFIGURATION_FILE_PATH)
    training_log_rows = read_training_log_rows(TRAINING_LOG_FILE_PATH)

    with st.sidebar:
      st.header("Run Configuration")
      if run_configuration:
        st.json(run_configuration["hyperparameters"], expanded=False)
        st.json(run_configuration["dataset_information"])
      else:
        st.warning("run_config.json not found for this run.")

    graphs_tab, logs_tab = st.tabs(["📈 Graphs & Metrics", "📄 Logs"])
    with graphs_tab:
      display_run_data(training_log_rows, run_configuration)
    with logs_tab:
      st.subheader("Console Logs")
      if CONSOLE_LOG_FILE_PATH.exists():
        with open(CONSOLE_LOG_FILE_PATH, "r") as file_handle:
          st.code(file_handle.read(), language="log", line_numbers=True)
      else:
        st.info("Console log not found.")
