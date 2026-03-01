import os

import hydra
import omegaconf


@hydra.main(config_path="configurations", config_name="train", version_base=None)
def train(configuration: omegaconf.DictConfig):
    dataset = hydra.utils.instantiate(
        configuration.dataset,
        files=list(
            map(
                lambda file: file.split(".")[0],
                filter(
                    lambda file: file.endswith(".jpg"),
                    os.listdir(configuration.dataset.path),
                ),
            )
        ),
        batch_size=4,
        num_workers=4,
        train_split=0.8,
        val_split=0.1,
        augmentations=None,
    )

    dataset.setup()

    model = hydra.utils.instantiate(configuration.model)

    train_dataloader = dataset.train_dataloader()
    val_dataloader = dataset.val_dataloader()
    test_dataloader = dataset.test_dataloader()

    output = model.forward(next(iter(train_dataloader)))

    print("[output]:", output)


if __name__ == "__main__":
    train()
