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
    )

    print("[#dataset]:", len(dataset))


if __name__ == "__main__":
    train()
