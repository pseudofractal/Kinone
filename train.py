import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime

import numpy as np

from src.core.losses import bce_with_logits
from src.core.metrics import accuracy_score, roc_auc_score
from src.core.models.resnet import resnet18
from src.core.optim import Adam
from src.core.schedulers import StepLR
from src.core.tensor import Tensor
from src.data.nih_cxr import DISEASES
from src.data.nih_datamodule import NIHDataModule

START_TIME = time.time()


def _console_log(message: str, level: str, indent: int):
  valid_levels = {"INFO", "ERROR", "DEBUG"}
  if level not in valid_levels:
    raise ValueError(f"Invalid log level: '{level}'. Must be one of {valid_levels}")

  elapsed_time = time.time() - START_TIME
  if elapsed_time < 60:
    time_string = f"{elapsed_time:.2f}S"
  elif elapsed_time < 3600:
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    time_string = f"{minutes}M:{seconds:.2f}S"
  elif elapsed_time < 86400:
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    time_string = f"{hours}H:{minutes}M:{seconds:.2f}S"
  else:
    days = int(elapsed_time // 86400)
    hours = int((elapsed_time % 86400) // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    time_string = f"{days}D:{hours}H:{minutes}M:{seconds:.2f}S"

  indentation = " " * (indent * 2)
  print(f"{indentation}○ [{level}] {time_string} ∘ {message}")


def log(message: str, level: str = "INFO", indent: int = 0):
  _console_log(message, level, indent)


def main():
  log("Starting Training")
  parser = argparse.ArgumentParser(description="Train a ResNet-18 model.")

  parser.add_argument(
    "--dataset", type=str, default="nih", help="The dataset to use (e.g., 'nih')."
  )
  parser.add_argument(
    "--epochs", type=int, default=10, help="Number of epochs to train for."
  )
  parser.add_argument(
    "--batch-size", type=int, default=32, help="Batch size for training."
  )
  parser.add_argument(
    "--lr", type=float, default=1e-3, help="Learning rate for the optimizer."
  )
  parser.add_argument(
    "--lr-step-size",
    type=int,
    default=5,
    help="Step size for the learning rate scheduler.",
  )
  parser.add_argument(
    "--lr-gamma", type=float, default=0.1, help="Gamma for the learning rate scheduler."
  )
  parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility."
  )
  parser.add_argument(
    "--load-checkpoint", type=str, help="Path to a saved checkpoint to load."
  )
  parser.add_argument(
    "--export-onnx",
    type=str,
    default=None,
    help="Path to save the exported ONNX model after training.",
  )

  temp_arguments, _ = parser.parse_known_args()
  if temp_arguments.dataset == "nih":
    parser = NIHDataModule.add_argparse_args(parser)
  arguments = parser.parse_args()

  log("Processed Arguments")

  if arguments.dataset == "nih":
    data_module = NIHDataModule(arguments)
  else:
    raise ValueError(f"Unsupported dataset: {arguments.dataset}")

  log("Loaded Dataset")

  data_module.setup("fit")
  train_dataloader = data_module.train_dataloader()
  validation_dataloader = data_module.val_dataloader()

  model = resnet18(num_classes=len(DISEASES))
  optimizer = Adam(model.parameters(), learning_rate=arguments.lr)
  scheduler = StepLR(
    optimizer, step_size=arguments.lr_step_size, gamma=arguments.lr_gamma
  )

  log("Model and Optimizer Initialized")

  if arguments.load_checkpoint:
    model.load_state_dict(np.load(arguments.load_checkpoint))
    log(f"Loaded model from {arguments.load_checkpoint}", indent=1)

  best_auc = 0.0
  best_checkpoint_path = None
  log_file = "training_log.json"
  if os.path.exists(log_file):
    os.remove(log_file)

  def _graceful_interrupt(sig, frame):
    ckpt = "checkpoints/interrupted_%d.npz" % int(time.time())
    np.save(ckpt, {n: p.data for n, p in model.named_parameters()})
    log("SIGINT Caught – saving weights and exiting", "DEBUG")
    sys.exit(130)

  signal.signal(signal.SIGINT, _graceful_interrupt)

  for epoch in range(arguments.epochs):
    log(f"Epoch: {epoch + 1}/{arguments.epochs}")
    model.set_to_training()
    for i, (images, labels) in enumerate(train_dataloader):
      image_tensor = Tensor(images, requires_grad=True)
      label_tensor = Tensor(labels)

      predictions = model(image_tensor)
      loss, grad = bce_with_logits(predictions, label_tensor)

      predictions.grad = grad
      predictions.backward()

      optimizer.step()
      optimizer.zero_grad()

      if i % 10 == 0:
        log(
          f"Batch {i + 1}/{len(train_dataloader.dataset) // train_dataloader.batch_size}, Loss: {loss:.4f}",
          indent=2,
        )

    model.set_to_evaluation()
    all_predictions = []
    all_labels = []
    for i, (images, labels) in enumerate(validation_dataloader):
      image_tensor = Tensor(images)
      predictions = model(image_tensor)
      all_predictions.append(predictions.data)
      all_labels.append(labels)

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    auc = roc_auc_score(all_labels, all_predictions, average="macro")
    accuracy = accuracy_score(all_labels, all_predictions > 0.5)

    log(f"Validation AUC: {auc:.4f}, Accuracy: {accuracy:.4f}", "DEBUG", indent=1)

    log_entry = {
      "epoch": epoch + 1,
      "loss": loss,
      "val_auc": auc,
      "lr": optimizer.learning_rate,
    }
    with open(log_file, "a") as f:
      f.write(json.dumps(log_entry) + "\n")

    if auc > best_auc:
      best_auc = auc
      log(f"New best validation AUC: {best_auc:.4f}", indent=3)

      if best_checkpoint_path and os.path.exists(best_checkpoint_path):
        os.remove(best_checkpoint_path)

      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      best_checkpoint_path = f"checkpoints/best_model_{timestamp}.npz"
      model_weights = {name: p.data for name, p in model.named_parameters()}
      np.savez(best_checkpoint_path, **model_weights)
      log(f"Model saved to {best_checkpoint_path}", indent=3)

    scheduler.step()

  log("Finished Training.")
  print("\n\n")
  log("Evaluating Model on Test Set")
  data_module.setup("test")
  test_dataloader = data_module.test_dataloader()
  model.set_to_evaluation()
  all_predictions = []
  all_labels = []
  for i, (images, labels) in enumerate(test_dataloader):
    image_tensor = Tensor(images)
    predictions = model(image_tensor)
    all_predictions.append(predictions.data)
    all_labels.append(labels)

  all_predictions = np.concatenate(all_predictions)
  all_labels = np.concatenate(all_labels)

  auc = roc_auc_score(all_labels, all_predictions, average="macro")

  log(f"Test AUC: {auc:.4f}")

  if arguments.export_onnx:
    from src.export_onnx import export_onnx

    log(f"Exporting to ONNX format at {arguments.export_onnx}")
    dummy_input = Tensor(np.zeros((1, 1, 224, 224), dtype=np.float32))
    out = model(dummy_input)
    export_onnx(out, arguments.export_onnx)
    log(f"ONNX model written to {arguments.export_onnx}")


if __name__ == "__main__":
  main()
