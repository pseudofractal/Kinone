import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.core.losses import binary_cross_entropy_with_logits
from src.core.metrics import accuracy_score, roc_auc_score
from src.core.models.resnet import resnet18
from src.core.optim import Adam
from src.core.schedulers import StepLR
from src.core.tensor import Tensor
from src.data.nih_cxr import DISEASES
from src.data.nih_datamodule import NIHDataModule


def train(args):
  INITIAL_START_TIME = time.time()
  LOG_FREQUENCY = 1
  CONSOLE_LOG_FILE_PATH = Path(args.console_log_file)
  STOP_SIGNAL_FILE_PATH = Path(args.stop_signal_file)
  run_config_file_path = Path(args.run_config_file)

  def termination_handler(_, __):
    log_message("Termination signal received. Exiting gracefully.", "INFO")
    STOP_SIGNAL_FILE_PATH.touch()
    sys.exit(0)

  signal.signal(signal.SIGINT, termination_handler)
  signal.signal(signal.SIGTERM, termination_handler)

  def log_message(message: str, level: str = "INFO", indent: int = 0):
    valid_levels = {"INFO", "ERROR", "DEBUG"}
    if level not in valid_levels:
      raise ValueError(f"Invalid log level: '{level}'. Must be one of {valid_levels}")
    elapsed_time_seconds = time.time() - INITIAL_START_TIME
    time_string = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_seconds))
    indentation = " " * (indent * 2)
    formatted_log_message = f"{indentation}○ [{level}] {time_string} ∘ {message}"
    print(formatted_log_message)
    if CONSOLE_LOG_FILE_PATH:
      with open(CONSOLE_LOG_FILE_PATH, "a") as file_handle:
        file_handle.write(formatted_log_message + "\n")

  if CONSOLE_LOG_FILE_PATH.exists():
    CONSOLE_LOG_FILE_PATH.unlink()

  log_message("Starting Training Script")
  log_message("Processed Command-Line Arguments")
  Path("checkpoints").mkdir(exist_ok=True)

  if args.dataset == "nih":
    data_module = NIHDataModule(args)
  else:
    raise ValueError(f"Unsupported dataset: {args.dataset}")

  log_message("Loaded Dataset Module")
  data_module.setup("fit")
  train_dataloader = data_module.train_dataloader()
  validation_dataloader = data_module.val_dataloader()

  dataset_information = {
    "training_samples": len(data_module.train_dataset),
    "validation_samples": len(data_module.val_dataset),
    "test_samples": len(data_module.test_dataset),
  }

  args_dict = vars(args).copy()
  for key, value in args_dict.items():
    if isinstance(value, Path):
      args_dict[key] = str(value)

  run_configuration = {
    "hyperparameters": args_dict,
    "dataset_information": dataset_information,
  }
  with open(run_config_file_path, "w") as file_handle:
    json.dump(run_configuration, file_handle, indent=2)

  train_dataset = train_dataloader.dataset
  positive_sample_counts = np.zeros(len(DISEASES), dtype=np.int64)
  for _, label_vector in train_dataset.samples:
    positive_sample_counts += label_vector
  negative_sample_counts = len(train_dataset) - positive_sample_counts
  positive_class_weight = (
    negative_sample_counts / np.clip(positive_sample_counts, 1, None)
  ).astype(np.float32)
  np.clip(positive_class_weight, 1.0, 20.0, out=positive_class_weight)
  log_message("Using capped positive class weights.", "DEBUG")

  model = resnet18(num_classes=len(DISEASES))
  optimizer = Adam(model.parameters(), learning_rate=args.lr)
  scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
  log_message("Model and Optimizer Initialized")

  if args.load_checkpoint:
    model.load_state_dict(np.load(args.load_checkpoint))
    log_message(f"Loaded model from {args.load_checkpoint}", indent=1)

  best_validation_auc = 0.0
  best_model_checkpoint_path = None
  training_log_file_path = Path(args.training_log_file)
  if training_log_file_path.exists():
    training_log_file_path.unlink()

  def immediate_interrupt_handler(signal_number, frame):
    log_message(
      "SIGINT Caught – saving final weights and exiting immediately.", "DEBUG"
    )
    interrupted_checkpoint_path = f"checkpoints/interrupted_{int(time.time())}.npz"
    model_weights = {
      name: parameter.data for name, parameter in model.named_parameters()
    }
    np.savez(interrupted_checkpoint_path, **model_weights)
    log_message(f"Model saved to {interrupted_checkpoint_path}", indent=1)
    sys.exit(130)

  signal.signal(signal.SIGINT, immediate_interrupt_handler)

  log_message(f"Logging progress to {training_log_file_path}", indent=1)
  log_message(f"Logging every {LOG_FREQUENCY} batches", indent=1)

  for epoch_index in range(args.epochs):
    epoch_start_time = time.time()
    log_message(f"Epoch: {epoch_index + 1}/{args.epochs}")
    model.set_to_training()
    total_training_loss = 0.0
    number_of_batches = 0

    for batch_index, (images, labels) in enumerate(train_dataloader):
      image_tensor = Tensor(images, requires_grad=True)
      label_tensor = Tensor(labels)
      predictions = model(image_tensor)
      loss_value, gradient_tensor = binary_cross_entropy_with_logits(
        predictions, label_tensor, positive_class_weight
      )
      predictions.backward(gradient_tensor)
      optimizer.step()
      optimizer.zero_grad()
      total_training_loss += loss_value
      number_of_batches += 1
      if batch_index % LOG_FREQUENCY == 0:
        log_message(f"Batch {batch_index + 1}/{len(train_dataloader.dataset) // train_dataloader.batch_size}, Loss: {loss_value:.4f}", indent=2)

    average_training_loss = (
      total_training_loss / number_of_batches if number_of_batches > 0 else 0
    )

    log_message(f"Average Training Loss: {average_training_loss:.4f}", indent=2)
    log_message("Starting Validation", indent=1)
    model.set_to_evaluation()
    all_validation_predictions = []
    all_validation_labels = []
    total_validation_loss = 0.0
    number_of_validation_batches = 0

    for images, labels in validation_dataloader:
      image_tensor = Tensor(images)
      label_tensor = Tensor(labels)
      predictions = model(image_tensor)
      loss_value, _ = binary_cross_entropy_with_logits(
        predictions, label_tensor, positive_class_weight
      )
      all_validation_predictions.append(predictions.data)
      all_validation_labels.append(labels)
      total_validation_loss += loss_value
      number_of_validation_batches += 1

    average_validation_loss = (
      total_validation_loss / number_of_validation_batches
      if number_of_validation_batches > 0
      else 0
    )
    all_validation_predictions = np.concatenate(all_validation_predictions)
    all_validation_labels = np.concatenate(all_validation_labels)

    macro_validation_auc = roc_auc_score(
      all_validation_labels, all_validation_predictions, average="macro"
    )
    validation_accuracy = accuracy_score(
      all_validation_labels, all_validation_predictions > 0.5
    )

    per_class_validation_auc = {}
    for class_index, disease_name in enumerate(DISEASES):
      class_labels = all_validation_labels[:, class_index]
      class_scores = all_validation_predictions[:, class_index]
      if np.any(class_labels) and not np.all(class_labels):
        per_class_validation_auc[disease_name] = roc_auc_score(
          class_labels, class_scores
        )

    log_message(
      f"Validation AUC: {macro_validation_auc:.4f}, Accuracy: {validation_accuracy:.4f}",
      "DEBUG",
      indent=1,
    )

    epoch_end_time = time.time()
    log_entry = {
      "epoch": epoch_index + 1,
      "average_training_loss": average_training_loss,
      "validation_loss": average_validation_loss,
      "validation_auc": macro_validation_auc,
      "validation_accuracy": validation_accuracy,
      "per_class_validation_auc": per_class_validation_auc,
      "learning_rate": optimizer.learning_rate,
      "epoch_duration_seconds": epoch_end_time - epoch_start_time,
      "total_elapsed_time_seconds": epoch_end_time - INITIAL_START_TIME,
    }
    with open(training_log_file_path, "a") as file_handle:
      file_handle.write(json.dumps(log_entry) + "\n")

    if macro_validation_auc > best_validation_auc:
      best_validation_auc = macro_validation_auc
      log_message(f"New best validation AUC: {best_validation_auc:.4f}", indent=3)

      if best_model_checkpoint_path and os.path.exists(best_model_checkpoint_path):
        os.remove(best_model_checkpoint_path)

      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      best_model_checkpoint_path = f"checkpoints/best_model_{timestamp}.npz"
      model_weights = {
        name: parameter.data for name, parameter in model.named_parameters()
      }
      np.savez(best_model_checkpoint_path, **model_weights)
      log_message(f"Model saved to {best_model_checkpoint_path}", indent=3)

    scheduler.step()

    if STOP_SIGNAL_FILE_PATH.exists():
      log_message("Stop signal file detected. Terminating training.", "INFO")
      STOP_SIGNAL_FILE_PATH.unlink()
      break

  log_message("Finished Training.")

  if args.export_onnx:
    from src.scripts.export_onnx import export_onnx

    log_message(f"Exporting to ONNX format at {args.export_onnx}")
    dummy_input = Tensor(np.zeros((1, 1, 224, 224), dtype=np.float32))
    output_tensor = model(dummy_input)
    export_onnx(output_tensor, args.export_onnx)
    log_message(f"ONNX model written to {args.export_onnx}")
