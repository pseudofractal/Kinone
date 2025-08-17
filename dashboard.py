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


BASE_ARGUMENT_PARSER = argparse.ArgumentParser(add_help=False)

BASE_ARGUMENT_PARSER = add_path_arguments(BASE_ARGUMENT_PARSER, load_config())
BASE_ARGUMENT_PARSER.add_argument("--training-pid", type=int, default=None)
COMMAND_LINE_ARGUMENTS, _ = BASE_ARGUMENT_PARSER.parse_known_args()

TRAINING_LOG_FILE_PATH = Path(COMMAND_LINE_ARGUMENTS.training_log_file)
CONSOLE_LOG_FILE_PATH = Path(COMMAND_LINE_ARGUMENTS.console_log_file)
RUN_CONFIGURATION_FILE_PATH = Path(COMMAND_LINE_ARGUMENTS.run_config_file)
STOP_SIGNAL_FILE_PATH = Path(COMMAND_LINE_ARGUMENTS.stop_signal_file)
TRAINING_PROCESS_IDENTIFIER = COMMAND_LINE_ARGUMENTS.training_pid

if STOP_SIGNAL_FILE_PATH.exists():
  STOP_SIGNAL_FILE_PATH.unlink()
  if "stop_signal_sent" in st.session_state:
    st.session_state.stop_signal_sent = False


def read_training_log_rows() -> list[dict]:
  if not TRAINING_LOG_FILE_PATH.exists():
    return []
  try:
    with open(TRAINING_LOG_FILE_PATH, "r") as log_file_handle:
      file_content = log_file_handle.read()
      if not file_content:
        return []
      return json.loads(file_content)
  except (json.JSONDecodeError, FileNotFoundError):
    return []


def load_run_configuration() -> dict | None:
  if not RUN_CONFIGURATION_FILE_PATH.exists():
    return None
  with open(RUN_CONFIGURATION_FILE_PATH, "r") as configuration_file_handle:
    return json.load(configuration_file_handle)


st.set_page_config(layout="wide", page_title="Kinone Training Dashboard")
st.title("Kinone Training Dashboard")

if "stop_signal_sent" not in st.session_state:
  st.session_state.stop_signal_sent = False

graceful_stop_button_column, immediate_stop_button_column = st.columns(2)
with graceful_stop_button_column:
  if st.button(
    "Stop after epoch",
    type="primary",
    help="Allow the current epoch to finish before stopping.",
  ):
    STOP_SIGNAL_FILE_PATH.touch()
    st.session_state.stop_signal_sent = True
    st.toast("Graceful stop signal sent.")
with immediate_stop_button_column:
  if st.button(
    "Immediate stop",
    type="secondary",
    help="Interrupt the training process immediately (SIGINT).",
  ):
    if TRAINING_PROCESS_IDENTIFIER:
      os.kill(TRAINING_PROCESS_IDENTIFIER, signal.SIGINT)
      st.toast("Immediate stop signal sent.")
    else:
      st.error("Training PID not provided")

run_configuration = load_run_configuration()
with st.sidebar:
  st.header("Run Configuration")
  if run_configuration:
    st.subheader("Hyperparameters")
    st.json(run_configuration["hyperparameters"], expanded=False)
    st.subheader("Dataset Information")
    st.json(run_configuration["dataset_information"])
  else:
    st.info("Waiting for run_config.json...")

main_content_placeholder = st.empty()

while True:
  training_log_rows = read_training_log_rows()

  with main_content_placeholder.container():
    if st.session_state.stop_signal_sent:
      st.warning(
        "Stop signal has been sent. Training will halt after the current epoch finishes."
      )

    if not training_log_rows:
      st.info("Waiting for first epoch to complete...")
    else:
      training_dataframe = pandas.DataFrame(training_log_rows)
      latest_epoch_data = training_dataframe.iloc[-1]
      best_area_under_curve_epoch_data = training_dataframe.loc[
        training_dataframe["validation_auc"].idxmax()
      ]

      if run_configuration:
        total_number_of_epochs = run_configuration["hyperparameters"].get(
          "epochs", latest_epoch_data["epoch"]
        )
      else:
        total_number_of_epochs = latest_epoch_data["epoch"]

      st.header("Live Metrics")
      st.progress(
        latest_epoch_data["epoch"] / total_number_of_epochs,
        text=f"Epoch {latest_epoch_data['epoch']} / {total_number_of_epochs}",
      )

      key_performance_indicator_columns = st.columns(5)

      area_under_curve_delta = (
        latest_epoch_data["validation_auc"]
        - best_area_under_curve_epoch_data["validation_auc"]
      )
      key_performance_indicator_columns[0].metric(
        label="Validation AUC",
        value=f"{latest_epoch_data['validation_auc']:.4f}",
        delta=f"{area_under_curve_delta:.4f} vs Best",
        help="The macro-averaged Area Under ROC Curve. Delta shows the difference from the best AUC achieved so far.",
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
        delta=f"{validation_loss_delta:.4f}"
        if validation_loss_delta is not None
        else None,
        delta_color="inverse",
        help="The average loss on the validation set. Lower is better.",
      )

      key_performance_indicator_columns[2].metric(
        label="Best Validation AUC",
        value=f"{best_area_under_curve_epoch_data['validation_auc']:.4f}",
        help=f"The best AUC was achieved at epoch {best_area_under_curve_epoch_data['epoch']}.",
      )

      key_performance_indicator_columns[3].metric(
        label="Last Epoch Duration",
        value=format_seconds_to_human_readable_string(
          latest_epoch_data["epoch_duration_seconds"]
        ),
        help="Time taken for the most recently completed epoch.",
      )

      average_epoch_duration_seconds = training_dataframe[
        "epoch_duration_seconds"
      ].mean()
      key_performance_indicator_columns[4].metric(
        label="Average Epoch Duration",
        value=format_seconds_to_human_readable_string(average_epoch_duration_seconds),
        help="The average time taken per epoch across the entire run.",
      )

      graphs_tab, logs_tab = st.tabs(["📈 Graphs", "📄 Logs"])

      with graphs_tab:
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
          use_container_width=True,
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
          use_container_width=True,
        )

        st.subheader("Per-Class Validation AUC (Latest Epoch)")
        per_class_area_under_curve_dataframe = pandas.DataFrame.from_dict(
          latest_epoch_data["per_class_validation_auc"], orient="index", columns=["AUC"]
        ).sort_values(by="AUC", ascending=True)
        st.bar_chart(
          per_class_area_under_curve_dataframe,
          horizontal=True,
          use_container_width=True,
        )

        st.subheader("Training Performance")
        timing_and_learning_rate_columns = st.columns(2)
        with timing_and_learning_rate_columns[0]:
          st.line_chart(
            training_dataframe.rename(
              columns={"epoch_duration_seconds": "Epoch Duration (s)"}
            ),
            x="epoch",
            y="Epoch Duration (s)",
            use_container_width=True,
          )
        with timing_and_learning_rate_columns[1]:
          st.line_chart(
            training_dataframe.rename(columns={"learning_rate": "Learning Rate"}),
            x="epoch",
            y="Learning Rate",
            use_container_width=True,
          )

      with logs_tab:
        st.subheader("Console Logs")
        if CONSOLE_LOG_FILE_PATH.exists():
          with open(CONSOLE_LOG_FILE_PATH, "r") as file_handle:
            log_lines = file_handle.readlines()
            st.code("".join(log_lines[-50:]), language="log", line_numbers=False)
            with st.expander("Show full console log"):
              st.code("".join(log_lines), language="log", line_numbers=True)
        else:
          st.info("Console log not found")

  time.sleep(5)
