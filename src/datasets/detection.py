import dataclasses
import json
import os
import typing

import lightning
import numpy
import PIL
import PIL.Image
import torch

import data.drivers.index as drivers
import src.datasets.detection as detection


@dataclasses.dataclass
class DetectionDatasetItem:
    image: torch.Tensor
    annotation: drivers.FrameAnnotation
    file_name: str


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        files: typing.List[str],
        augmentations: typing.Optional[typing.Callable] = None,
    ):
        self.path = path
        self.files = files
        self.augmentations = augmentations

        for file in files:
            assert os.path.exists(os.path.join(path, f"{file}.jpg")), (
                f"Image file {file} does not exist in {path}"
            )
            assert os.path.exists(os.path.join(path, f"{file}.json")), (
                f"Annotation file {file} does not exist in {path}"
            )

    def __getitem__(self, index: int) -> DetectionDatasetItem:
        file_name = self.files[index]
        image_path = os.path.join(self.path, f"{file_name}.jpg")
        annotation_path = os.path.join(self.path, f"{file_name}.json")

        image = torch.tensor(numpy.array(PIL.Image.open(image_path).convert("RGB")))

        with open(annotation_path, "r") as f:
            annotation = json.load(f)

        if self.augmentations is not None:
            image = self.augmentations(image)

        return DetectionDatasetItem(
            image=image,
            annotation=annotation,
            file_name=file_name,
        )

    def __len__(self) -> int:
        return len(self.files)


@dataclasses.dataclass
class DetectionBatch:
    # (size, 3, H, W)
    images: torch.Tensor
    annotations: typing.List[drivers.FrameAnnotation]
    file_names: typing.List[str]


def detection_collate_fn(batch: typing.List[DetectionDatasetItem]) -> DetectionBatch:
    images = torch.stack([item.image for item in batch])
    annotations = [item.annotation for item in batch]
    file_names = [item.file_name for item in batch]

    return DetectionBatch(
        images=images,
        annotations=annotations,
        file_names=file_names,
    )


class DetectionDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        path: str,
        files: typing.List[str],
        batch_size: int,
        num_workers: int,
        train_split: float,
        val_split: float,
        augmentations: typing.Optional[typing.Callable] = None,
    ):
        """Initialize data module.

        Args:
            data_path: Path to dataset directory
            files: List of file base names (without extensions)
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            augmentations: Optional augmentation pipeline
        """
        super().__init__()

        self.data_path = path
        self.file_list = files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.augmentations = augmentations

        self.train_dataset: typing.Optional[torch.utils.data.Subset] = None
        self.val_dataset: typing.Optional[torch.utils.data.Subset] = None
        self.test_dataset: typing.Optional[torch.utils.data.Subset] = None

    def setup(self, stage: typing.Optional[str] = None) -> None:
        """Setup datasets for train, val, and test stages.

        Args:
            stage: Stage of training (fit, validate, test, or predict)
        """
        if stage == "fit" or stage is None:
            full_dataset = detection.DetectionDataset(
                path=self.data_path,
                files=self.file_list,
                augmentations=self.augmentations,
            )

            total_size = len(full_dataset)
            train_size = int(total_size * self.train_split)
            val_size = int(total_size * self.val_split)
            test_size = total_size - train_size - val_size

            self.train_dataset, self.val_dataset, self.test_dataset = (
                torch.utils.data.random_split(
                    full_dataset,
                    [train_size, val_size, test_size],
                    generator=torch.Generator().manual_seed(42),
                )
            )

        if stage == "test" or stage is None:
            if self.test_dataset is None:
                full_dataset = detection.DetectionDataset(
                    path=self.data_path,
                    files=self.file_list,
                    augmentations=self.augmentations,
                )
                total_size = len(full_dataset)
                train_size = int(total_size * self.train_split)
                val_size = int(total_size * self.val_split)
                test_size = total_size - train_size - val_size

                _, _, self.test_dataset = torch.utils.data.random_split(
                    full_dataset,
                    [train_size, val_size, test_size],
                    generator=torch.Generator().manual_seed(42),
                )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=detection_collate_fn,
        )

    def val_dataloader(self) -> typing.Optional[torch.utils.data.DataLoader]:
        if self.val_dataset is None:
            return None

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=detection_collate_fn,
        )

    def test_dataloader(self) -> typing.Optional[torch.utils.data.DataLoader]:
        if self.test_dataset is None:
            return None

        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=detection_collate_fn,
        )

    def predict_dataloader(self) -> typing.Optional[torch.utils.data.DataLoader]:
        return self.test_dataloader()
