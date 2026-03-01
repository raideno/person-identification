import abc
from dataclasses import dataclass, field, asdict
from typing import List
from uuid import uuid4


@dataclass
class BoundingBox:
    """Bounding box annotation with normalized coordinates."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class Detection:
    """Single person detection in an image."""
    class_name: str
    bbox: BoundingBox


@dataclass
class FrameAnnotation:
    """Annotation for a single frame/image."""
    frame_id: str
    sequence_id: str
    dataset: str
    image_width: int
    image_height: int
    detections: List[Detection] = field(default_factory=list)

    def to_dict(self):
        """Convert annotation to dictionary for JSON serialization."""
        return {
            "frame_id": self.frame_id,
            "sequence_id": self.sequence_id,
            "dataset": self.dataset,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "detections": [
                {
                    "class_name": det.class_name,
                    "bbox": asdict(det.bbox),
                }
                for det in self.detections
            ],
        }


class AbstractDriver(abc.ABC):
    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def process(self, **kwargs):
        pass