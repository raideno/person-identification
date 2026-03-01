import csv
import json

import path

from index import AbstractDriver, BoundingBox, Detection, FrameAnnotation


class PeopleDetectionDriver(AbstractDriver):
    """Driver for processing Roboflow people-detection dataset.

    Converts CSV annotations to standardized format with UUID-based naming.
    Each image is treated as its own sequence.
    """

    def __init__(self, dataset_path: str, output_path: str):
        """Initialize driver with dataset and output paths.

        Args:
            dataset_path: Path to the root people_detection dataset folder
            output_path: Path to output directory for processed data
        """
        self.dataset_path = path.Path(dataset_path)
        self.output_path = path.Path(output_path)
        self.dataset_name = "people-detection"

    def process(self, **kwargs) -> None:
        """Process the entire dataset.

        Reads annotations from train/valid/test splits and writes
        standardized output to output_path.
        """
        self.output_path.mkdir(parents=True, exist_ok=True)

        sequence_counter = 1
        splits = ["train", "valid", "test"]
        for split in splits:
            split_path = self.dataset_path / split / split
            if not split_path.exists():
                continue
            sequence_counter = self._process_split(split_path, split, sequence_counter)

    def _process_split(
        self, split_path: path.Path, split_name: str, sequence_counter: int
    ) -> int:
        """Process a single dataset split (train/valid/test).

        Args:
            split_path: Path to the split directory
            split_name: Name of the split
            sequence_counter: Current sequence counter

        Returns:
            Updated sequence counter
        """
        annotations_file = split_path / "_annotations.csv"
        if not annotations_file.exists():
            print(f"No annotations found for {split_name}")
            return sequence_counter

        # Group annotations by filename
        annotations_by_image = {}
        with open(annotations_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"]
                if filename not in annotations_by_image:
                    annotations_by_image[filename] = {
                        "width": int(row["width"]),
                        "height": int(row["height"]),
                        "detections": [],
                    }
                annotations_by_image[filename]["detections"].append(
                    {
                        "class": row["class"],
                        "xmin": float(row["xmin"]),
                        "ymin": float(row["ymin"]),
                        "xmax": float(row["xmax"]),
                        "ymax": float(row["ymax"]),
                    }
                )

        # Process each image with its annotations
        for filename, anno_data in annotations_by_image.items():
            sequence_counter = self._process_image(
                split_path, filename, anno_data, split_name, sequence_counter
            )

        print(f"Processed {split_name}: {len(annotations_by_image)} images")
        return sequence_counter

    def _process_image(
        self,
        split_path: path.Path,
        filename: str,
        anno_data: dict,
        split_name: str,
        sequence_counter: int,
    ) -> int:
        """Process a single image and write annotation.

        Args:
            split_path: Path to the split directory
            filename: Image filename
            anno_data: Dictionary with width, height, detections
            split_name: Name of the split
            sequence_counter: Current sequence counter

        Returns:
            Updated sequence counter
        """
        # Use simple counters instead of UUIDs
        sequence_id = sequence_counter
        frame_id = 1  # Each sequence has one frame

        # Create annotation object
        detections = [
            Detection(
                class_name=det["class"],
                bbox=BoundingBox(
                    x_min=det["xmin"],
                    y_min=det["ymin"],
                    x_max=det["xmax"],
                    y_max=det["ymax"],
                ),
            )
            for det in anno_data["detections"]
        ]

        annotation = FrameAnnotation(
            frame_id=frame_id,
            sequence_id=sequence_id,
            dataset=self.dataset_name,
            image_width=anno_data["width"],
            image_height=anno_data["height"],
            detections=detections,
        )

        # Create output filenames using convention: dataset-sequence-frame
        base_name = f"{self.dataset_name}-{sequence_id}-{frame_id}"

        # Copy image to output with new name
        src_image = split_path / filename
        dst_image = self.output_path / f"{base_name}.jpg"

        # Read and write image
        if src_image.exists():
            with open(src_image, "rb") as src, open(dst_image, "wb") as dst:
                dst.write(src.read())

        # Write annotation as JSON
        anno_file = self.output_path / f"{base_name}.json"
        with open(anno_file, "w") as f:
            json.dump(annotation.to_dict(), f, indent=2)

        return sequence_counter + 1
