import dataclasses
import typing

import lightning
import torch

import src.datasets as datasets


@dataclasses.dataclass
class ForwardOutput:
    batch_index: int
    # Each dict is a prediction for one image
    # It contains the following keys: boxes, scores, labels
    predictions: typing.List[typing.Dict[str, torch.Tensor]]


class DetectionModel(lightning.LightningModule):
    # TODO: add accuracy monitoring
    def __init__(self):
        super().__init__()

    def loss(self, batch: datasets.DetectionBatch, forward: ForwardOutput):
        return torch.tensor(0.0)

    def forward(self, batch: datasets.DetectionBatch):
        batch_size = len(batch.file_names)

        return ForwardOutput(
            batch_index=0,
            predictions=[
                {
                    "boxes": torch.tensor([]),
                    "scores": torch.tensor([]),
                    "labels": torch.tensor([]),
                }
            ]
            * batch_size,
        )

    def training_step(self, batch: datasets.DetectionBatch, batch_idx: int):
        forward = self.forward(batch)
        loss = self.loss(batch, forward)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: datasets.DetectionBatch, batch_idx: int):
        forward = self.forward(batch)
        loss = self.loss(batch, forward)

        self.log("val_loss", loss)

        return loss

    def test_step(self, batch: datasets.DetectionBatch, batch_idx: int):
        forward = self.forward(batch)
        loss = self.loss(batch, forward)

        self.log("test_loss", loss)

        return loss
