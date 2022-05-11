# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Potsdam datamodule."""

from typing import Any, Dict, Optional, Callable

import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from ..datasets import Potsdam2D
from .utils import dataset_split


class Potsdam2DDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the Potsdam2D dataset.

    Uses the train/test splits from the dataset.

    .. versionadded:: 0.2
    """

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        patch_size: int = 256,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for Potsdam2D based DataLoaders.

        Args:
            root_dir: The ``root`` argument to pass to the Potsdam2D Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            patch_size: The size of each patch in pixels (test patches will be 1.5 times
                this size)
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct

    def center_crop(
        self, size: int = 512
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a center crop transform on a single sample.

        Args:
            size: output image size

        Returns:
            function to perform center crop
        """

        def center_crop_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            _, height, width = sample["image"].shape

            y1 = (height - size) // 2
            x1 = (width - size) // 2
            sample["image"] = sample["image"][:, y1: y1 + size, x1: x1 + size]
            sample["mask"] = sample["mask"][y1: y1 + size, x1: x1 + size]

            return sample

        return center_crop_inner

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0
        return sample

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        transforms = Compose([
            self.center_crop(self.patch_size),
            self.preprocess
        ])

        dataset = Potsdam2D(self.root_dir, "train", transforms=transforms)

        self.train_dataset: Dataset[Any]
        self.val_dataset: Dataset[Any]

        if self.val_split_pct > 0.0:
            self.train_dataset, self.val_dataset, _ = dataset_split(
                dataset, val_pct=self.val_split_pct, test_pct=0.0
            )
        else:
            self.train_dataset = dataset
            self.val_dataset = dataset

        self.test_dataset = Potsdam2D(self.root_dir, "test", transforms=transforms)

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
