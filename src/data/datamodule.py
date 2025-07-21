import argparse
from abc import ABC, abstractmethod

from src.data.nih_cxr import DataLoader


class DataModule(ABC):
    """
    An abstract base class for creating dataset-specific data modules.

    A DataModule encapsulates all the steps needed to process data:
    downloading, splitting, and creating data loaders.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @abstractmethod
    def setup(self, stage: str | None = None) -> None:
        """Prepare the data for the given stage ('fit' or 'test')."""
        pass

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Return the DataLoader for the training set."""
        pass

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Return the DataLoader for the validation set."""
        pass

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """Return the DataLoader for the test set."""
        pass

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add dataset-specific arguments to the parser."""
        return parser
