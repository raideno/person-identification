import os
import typing


class DetectionDataset:
    # pass on augmentations array
    def __init__(self, path: str, files: typing.List[str]):
        self.path = path
        self.files = files

        for file in files:
            assert os.path.exists(os.path.join(path, f"{file}.jpg")), (
                f"Image file {file} does not exist in {path}"
            )
            assert os.path.exists(os.path.join(path, f"{file}.json")), (
                f"Annotation file {file} does not exist in {path}"
            )

    def __getitem__(self, index: int):
        pass

    def __len__(self):
        return len(self.files)
